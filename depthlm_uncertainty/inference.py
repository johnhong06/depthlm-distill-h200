"""DepthLM 모델 로드 + 생성. 공식 `eval.py`의 Pixtral 경로를 그대로 따른다.

공식 eval.py와 맞춘 점:
  - 메시지 구성은 `convert_example_pixtral(..., image_before_text=True)` 와 동일
    (system 메시지 없음, 이미지 먼저 → 텍스트)
  - `{"type": "text", "content": problem}` 키를 그대로 쓴다. Pixtral chat template이
    `content` / `text` 둘 다 받도록 되어 있음을 확인했다.
  - greedy 는 do_sample=False, top_p=None, top_k=None

바꾼 점:
  - flash_attention_2 → sdpa (`~/venv/main`에 flash-attn 미설치, Blackwell sm_120)
  - **vision 도 eager → sdpa.** 공식 코드는 vision_config="eager" 인데, sdpa 로 바꾸면
    batch 상한이 2 → 8, 처리량이 0.81 → 1.07 sample/s 로 올라간다. 같은 200 샘플에서
    δ₁ 0.8250 으로 동일함을 확인했다 (NOTES D-12). 둘 다 exact attention 이고
    커널만 다르다.
  - output_logits=True 로 토큰 uncertainty용 원 logits를 함께 받는다
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .metrics import parse_depth_answer
from .token_uncertainty import token_uncertainty_from_logits

logger = logging.getLogger(__name__)


@dataclass
class Generation:
    text: str
    depth: float | None
    token_stats: dict[str, float] | None = None


class DepthLMRunner:
    def __init__(
        self,
        model_path: str,
        attn_text: str = "sdpa",
        attn_vision: str = "sdpa",
        max_new_tokens: int = 128,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        logger.info("loading processor from %s", model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        logger.info("loading DepthLM (pixtral architecture) ...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation={
                "text_config": attn_text,
                "vision_config": attn_vision,
            },
            device_map="cuda:0",
        )
        self.model.eval()
        self.dtype = dtype

    # -- 입력 구성 ---------------------------------------------------------

    @staticmethod
    def _messages(image: Image.Image, problem: str) -> list[dict]:
        """공식 convert_example_pixtral(image_before_text=True) 와 동일."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "content": problem},
                ],
            }
        ]

    def _prepare(self, images: list[Image.Image], problem: str):
        chat = [self._messages(img, problem) for img in images]
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return inputs.to("cuda", dtype=self.dtype)

    # -- 생성 --------------------------------------------------------------

    @torch.no_grad()
    def generate_greedy(
        self,
        images: list[Image.Image],
        problem: str,
        with_token_stats: bool = False,
    ) -> list[Generation]:
        """결정적(greedy) 생성. with_token_stats=True면 U_token도 함께 계산한다."""
        inputs = self._prepare(images, problem)
        in_len = inputs["input_ids"].shape[1]

        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            return_dict_in_generate=True,
            output_logits=with_token_stats,
        )
        new_ids = out.sequences[:, in_len:]
        texts = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results: list[Generation] = []
        tok = self.processor.tokenizer
        for b, text in enumerate(texts):
            stats = None
            if with_token_stats:
                ids_b = new_ids[b]
                # eos 이후 padding 토큰은 통계에서 제외
                keep = int((ids_b != tok.pad_token_id).sum().item()) if tok.pad_token_id is not None else ids_b.shape[0]
                keep = max(keep, 1)
                token_strings = [
                    tok.decode([int(i)], skip_special_tokens=False)
                    for i in ids_b[:keep]
                ]
                logits_seq = [out.logits[t][b] for t in range(min(keep, len(out.logits)))]
                stats = token_uncertainty_from_logits(
                    logits_seq, ids_b[:keep], token_strings
                )
            results.append(Generation(text=text, depth=parse_depth_answer(text), token_stats=stats))

        del out, inputs
        return results

    @torch.no_grad()
    def generate_greedy_pertoken(
        self, images: list[Image.Image], problem: str
    ) -> list[dict]:
        """greedy 생성 + **모든 토큰**의 원시 통계를 그대로 반환.

        `generate_greedy` 는 숫자 span 만 집계해서 돌려주므로, "숫자 안에서 어느
        토큰이 신뢰도 정보를 담는가"(H2)를 사후에 물을 수 없다. 여기서는 집계하지
        않고 토큰마다 (위치, 문자열, id, logprob, entropy) 를 그대로 남긴다.
        재추론 없이 어떤 집계 방식이든 다시 계산할 수 있게 하는 것이 목적이다.

        `scores` 가 아니라 `logits` 를 쓴다 — NOTES D-6.
        """
        inputs = self._prepare(images, problem)
        in_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        new_ids = out.sequences[:, in_len:]
        texts = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        tok = self.processor.tokenizer
        results: list[dict] = []
        for b, text in enumerate(texts):
            ids_b = new_ids[b]
            keep = (
                int((ids_b != tok.pad_token_id).sum().item())
                if tok.pad_token_id is not None
                else ids_b.shape[0]
            )
            keep = max(1, min(keep, len(out.logits)))
            toks = []
            for t in range(keep):
                logits = out.logits[t][b].to(torch.float32)
                logprobs = torch.log_softmax(logits, dim=-1)
                tid = int(ids_b[t].item())
                lp = float(logprobs[tid].item())
                pr = torch.exp(logprobs)
                ent = float(-(pr * logprobs).sum().item())
                top = torch.topk(logprobs, 5)
                toks.append({
                    "pos": t,
                    "token_id": tid,
                    "token_str": tok.decode([tid], skip_special_tokens=False),
                    "logprob": lp,
                    "entropy": ent,
                    "top5_ids": [int(x) for x in top.indices.tolist()],
                    "top5_logprobs": [float(x) for x in top.values.tolist()],
                })
            results.append({"text": text, "tokens": toks})
        del out, inputs
        return results

    @torch.no_grad()
    def generate_sampled(
        self,
        image: Image.Image,
        problem: str,
        num_samples: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        chunk: int = 0,
    ) -> list[Generation]:
        """같은 입력을 M회 샘플링. prefix가 동일하므로 prefill은 1회만 돈다.

        num_return_sequences=M 은 prefill 후 KV 캐시를 M배로 확장한다. VRAM이
        빠듯하면 chunk 로 나눠 돈다 (prefill 횟수는 늘지만 죽지는 않는다).
        """
        chunk = chunk or num_samples

        results: list[Generation] = []
        done, ci = 0, 0
        while done < num_samples:
            n = min(chunk, num_samples - done)
            # 청크마다 따로 시드를 준다. manual_seed 를 루프 밖에서 한 번만 부르면
            # 청크들이 RNG 스트림을 순차 소비해서, chunk 크기가 바뀌면 (OOM 백오프 등)
            # 뽑히는 샘플 자체가 달라진다 — 재개 시 재현이 깨진다.
            if seed is not None:
                torch.manual_seed(seed + ci)
            inputs = self._prepare([image], problem)
            in_len = inputs["input_ids"].shape[1]
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
                num_return_sequences=n,
                return_dict_in_generate=True,
            )
            texts = self.processor.batch_decode(
                out.sequences[:, in_len:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            results.extend(Generation(text=t, depth=parse_depth_answer(t)) for t in texts)
            del out, inputs
            done += n
            ci += 1
        return results
