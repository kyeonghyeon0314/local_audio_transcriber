# Local Multimodal Transcriber

녹음 파일과 발표자료 PDF를 넣고 실행하면, 로컬 PC의 **Gemma 4 E4B IT**가 **원본 오디오와 PDF를 직접 참고해** 화자를 구분한 전사 TXT를 만들어 줍니다.
Gemini에 녹음과 PDF를 올려 전사하던 작업을 클라우드 없이 PC에서 처리하는 것이 목표입니다.

- Whisper 같은 별도 STT를 쓰지 않습니다. 잘라낸 **원본 파형을 그대로** 멀티모달 모델에 넣습니다.
- PDF는 요약하지 않고, 기술용어·약어·고유명사를 바로잡는 **근거 자료**로만 씁니다.
- 한국어와 영어가 섞인 발화를 번역하지 않고 **말한 그대로** 남깁니다.

```
[00:03:21] Speaker 1:
여기서는 MPCC를 사용했습니다.
[00:03:28] Speaker 2:
그러면 lateral acceleration 제한은 얼마인가요?
[웃음]
```

## 요구 사항

- Windows 10/11
- NVIDIA GPU, VRAM 8GB 이상 권장 (RTX 5060 8GB 기준으로 설정됨). GPU가 없으면 CPU로 동작하지만 매우 느립니다.
- Python 3.10 ~ 3.13 ([python.org](https://www.python.org/downloads/)에서 설치할 때 "Add python.exe to PATH" 체크)
- Hugging Face 계정 (Gemma 라이선스 동의용)
- 디스크 여유 공간 30GB 이상 권장 (PyTorch와 모델 파일)

## 설치 (최초 1회)

1. 이 저장소를 내려받습니다 (Code → Download ZIP 후 압축 해제, 또는 `git clone`).
2. [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) 페이지에서 라이선스에 동의합니다.
3. **`setup.bat`** 을 더블클릭합니다.
   - 가상환경(`.venv`), PyTorch(CUDA 12.8), 필요한 패키지를 설치합니다.
   - 마지막에 Hugging Face 토큰을 묻습니다. [토큰 페이지](https://huggingface.co/settings/tokens)에서 Read 토큰을 만들어 붙여 넣으세요.

## 사용법

**방법 1: input 폴더**

1. `input` 폴더에 녹음 파일(wav, mp3, m4a 등)과 발표자료 PDF를 넣습니다.
2. **`run.bat`** 을 더블클릭합니다.
3. 끝나면 `output` 폴더가 열리고 `녹음파일명_transcript.txt`가 생성되어 있습니다.

**방법 2: 끌어다 놓기**

녹음 파일과 PDF를 함께 선택해 `run.bat` 아이콘 위에 끌어다 놓습니다.

참고:
- 처음 실행할 때는 모델 파일(십수 GB)을 내려받으므로 오래 걸립니다.
- 녹음 여러 개를 한 번에 넣어도 됩니다. 녹음과 **파일명이 같은 PDF**가 있으면 그 PDF를, 없으면 넣은 PDF 전체를 참고합니다.
- PDF 없이 녹음만 넣어도 동작합니다.
- 처리 중에는 구간마다 결과 파일을 갱신합니다. 중간에 창을 닫아도 그때까지의 결과는 남습니다.
- 같은 이름의 결과가 이미 있으면 덮어쓰지 않고 `_transcript_2.txt`처럼 새 이름으로 저장합니다.

## 동작 방식

```
recording.m4a ─→ 16kHz mono 변환 ─→ 25초 구간(5초 겹침)으로 분할
presentation.pdf ─→ 페이지 텍스트 추출 (+ 글자가 거의 없는 슬라이드는 이미지로)
                                  │
            [고정 프롬프트 + PDF 문맥 + 원본 오디오 구간 + 직전 전사 끝부분]
                                  ↓
                         Gemma 4 E4B IT (로컬)
                                  ↓
          겹침 구간 중복 제거 → 화자/시간 정리 → recording_transcript.txt
```

- Gemma 4는 오디오를 한 번에 최대 30초까지 받기 때문에 긴 녹음은 자동으로 나눕니다.
- 각 구간에는 직전 전사의 끝부분과 지금까지 나온 화자 번호를 함께 넘겨 **Speaker 번호가 구간마다 바뀌지 않도록** 합니다.
- 겹치는 5초 동안 이미 전사된 발화는 다음 구간 결과에서 자동으로 제거합니다.
- 각 구간의 모델 원본 출력은 `output/logs/*_chunks.log`에 저장됩니다. 품질을 비교하거나 프롬프트를 조정할 때 참고하세요.

## 설정

- **`prompt.txt`**: 모델에 매번 주는 고정 지시문입니다. 코드를 고치지 않고 이 파일만 수정해도 됩니다.
- **`config.yaml`**: 기본값 그대로 쓰면 됩니다. 자주 바꿀 만한 항목은 다음과 같습니다.

| 항목 | 기본값 | 설명 |
|---|---|---|
| `model` | `google/gemma-4-E4B-it` | Hugging Face 모델 ID 또는 로컬 폴더 경로 |
| `quantization` | `4bit` | `4bit` / `8bit` / `none` |
| `max_gpu_memory` | `7GiB` | 넘치는 부분은 CPU RAM으로 offload |
| `output.timestamp` | `true` | `[00:03:21]` 시간 표시 |
| `audio.chunk_length` / `overlap` | `25` / `5` | 구간 길이와 겹침(초). 구간은 최대 30초 |
| `pdf.mode` | `auto` | `auto` / `text` / `images` / `off` |

## 테스트용 옵션

`run.bat` 대신 명령 프롬프트에서 직접 실행할 수 있습니다.

```bat
.venv\Scripts\python.exe run.py --max-chunks 4
.venv\Scripts\python.exe run.py --no-pdf
.venv\Scripts\python.exe run.py --dry-run
.venv\Scripts\python.exe run.py 녹음.m4a 자료.pdf --no-open
```

- `--max-chunks 4`: 앞 4개 구간(약 1분)만 빠르게 확인
- `--no-pdf`: PDF 없이 오디오만 사용 (PDF 효과 비교용)
- `--dry-run`: 모델 없이 입력 파일, 구간 분할, 프롬프트만 확인
- 파일 경로를 직접 지정할 수도 있습니다.

## 문제 해결

- **모델을 불러오지 못했습니다**: Gemma 라이선스 동의와 `setup.bat`의 Hugging Face 로그인을 했는지 확인하세요. 다시 로그인하려면 `.venv\Scripts\hf.exe auth login`을 실행합니다.
- **CUDA out of memory**: 다른 GPU 프로그램(게임, 브라우저 하드웨어 가속 등)을 닫고 다시 실행하세요. 4bit 모드에서는 모델의 레이어별 임베딩(PLE, 수 GB)만 CPU RAM에 두고 나머지는 GPU에 올리므로 RAM은 16GB 이상을 권장합니다.
- **CUDA 사용 가능: False**: NVIDIA 드라이버를 최신으로 업데이트한 뒤 `setup.bat`을 다시 실행하세요.
- 오류 상세 내용은 `output/logs/run_*.log`에 저장됩니다.

## 개발 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| Phase 1 | Raw Audio → Gemma 4 E4B 전사 | 구현됨 (`--no-pdf`로 확인) |
| Phase 2 | Audio + PDF 동시 입력 | 구현됨 |
| Phase 3 | 긴 녹음 자동 분할/겹침/병합 | 구현됨 |
| Phase 4 | 구간 간 화자 번호 유지 | 프롬프트 문맥 전달 방식으로 구현. 필요하면 speaker embedding 보조 기능 검토 |
| Phase 5 | 더블클릭 실행, 진행률, 오류 로그, 결과 폴더 열기 | `run.bat`으로 구현. `.exe` 패키징은 추후 |
| Phase 6 | E4B ↔ 12B 비교 | `config.yaml`의 `model`만 바꿔 비교 가능 |

## 개발자용

```bash
pip install pytest numpy pyyaml
python -m pytest tests
```
