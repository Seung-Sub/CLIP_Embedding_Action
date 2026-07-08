"""청크 표현 변환 — time(원시 시계열) / dct(FAST식 주파수 계수, arXiv 2501.09747).

※ 상태: DCT는 연구 보류 (기본 config는 전부 time, LIBERO 무기여 확인됨). 코드는 유지
   — 재개 시 configs/phase1_libero.yaml의 data.chunk_repr: dct 한 줄로 활성화되고,
   phase2·평가·롤아웃은 phase1 ckpt의 "chunk_repr" 키를 통해 자동 추종한다.

DCT-II 정규직교(norm='ortho')를 시간축(-2)에 적용: (..., T, D) → (..., T, D).
직교변환이라 MSE/R²는 보존되고, L1 손실의 기하만 바뀐다 — GT 고주파 계수가
거의 0이므로 모델이 저주파(궤적 형태) 우선으로 학습된다.

규약: 정규화(a_mean/a_std, 시간영역) 이후에 to_repr, 역정규화 직전에 from_repr.
phase1 체크포인트의 "chunk_repr" 키가 유일한 진실 — phase2·평가·롤아웃 전부
이 키를 읽어 동일 변환을 적용한다.
"""
import numpy as np
from scipy.fft import dct, idct

KINDS = ("time", "dct")


def to_repr(chunks, kind):
    """(..., T, D) 정규화 청크 → 표현공간."""
    if kind == "time":
        return chunks
    if kind == "dct":
        return dct(chunks, type=2, norm="ortho", axis=-2).astype(chunks.dtype)
    raise ValueError(f"chunk_repr={kind!r} (지원: {KINDS})")


def from_repr(chunks, kind):
    """표현공간 → (..., T, D) 정규화 청크."""
    if kind == "time":
        return chunks
    if kind == "dct":
        return idct(chunks, type=2, norm="ortho", axis=-2).astype(chunks.dtype)
    raise ValueError(f"chunk_repr={kind!r} (지원: {KINDS})")
