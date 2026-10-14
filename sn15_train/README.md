# sn15_train: 화풍 LoRA A/B 학습 실행 키트

배경과 근거는 [../SENSENOVA_8STEP_LORA_NOTES.md](../SENSENOVA_8STEP_LORA_NOTES.md)를 보세요.

> **결과 (2026-10-14)**: A/B 학습을 마쳤고 **B를 채택**했습니다. A는 8-step 출력이 흐릿하고 반투명해졌습니다. 자세한 내용은 노트의 "결과 요약"을 보세요. 이미지 생성은 루트의 [`inference.py`](../inference.py)로 합니다.

| 변형 | 학습 | 추론과 샘플 |
|---|---|---|
| **A** | 베이스 + 8-step V2(fused, 고정) 위에서 화풍 LoRA 학습 | 베이스 + 8-step V2 + A, 8 steps / cfg 1 |
| **B** | 베이스 위에서만 화풍 LoRA 학습 | 베이스 + 8-step V2 + B, 8 steps / cfg 1 |

두 변형 모두 같은 데이터, 같은 설정을 씁니다(트리거 없음, 560장 전부, rank 32, lr 1e-4, σ≥0.25). **다른 점은 학습할 때 8-step LoRA를 합치느냐 하나뿐입니다.**

## 학습 설정 (`train.sh`)
| 항목 | 값 | 바꾸는 곳 |
|---|---|---|
| 총 step | **10,080** (560장 × 18 epoch, 배치 1) | `train.sh`의 `--num_epochs` (총 step = 560 × epoch) |
| 체크포인트 | **500 step마다** → `step-500` … `step-10000`, 마지막 `step-10080` | `--save_steps` |
| 샘플 생성 | **체크포인트를 저장할 때마다**, 그리고 학습 시작 전(`step-0`)에 한 번 | `--sample_*` 인자 (아래) |
| 학습률 / rank | 1e-4 / 32 | `--learning_rate`, `--lora_rank` |
| 예상 시간 | 학습 약 3.4시간 + 샘플 약 15분 ≈ **변형당 약 3.5–4시간, A+B 약 7.5시간** `[추정]` | 실제 속도는 시험 학습으로 확인 |

파일을 고치지 않고 덮어쓰려면 인자를 뒤에 붙이면 됩니다. 예:
```bash
bash train.sh A --num_epochs 9 --learning_rate 5e-5
```

### 학습 중 샘플 생성 (`train_sn15.py`)
- `train_sn15.py`는 공식 `train.py`와 같은 학습을 합니다. 학습 코드는 그대로 불러와 쓰고, 다른 점은 체크포인트 저장 직후 샘플을 그리는 것뿐입니다.
- 학습 중인 가중치로 **같은 프로세스에서** 바로 그립니다. GPU에 모델을 하나 더 올리지 않습니다.
- 그리는 방법:
  - **A**: 이미 8-step이 합쳐져 있으므로 그대로 그립니다.
  - **B**: 샘플링하는 동안에만 8-step LoRA를 forward hook으로 얹고, 끝나면 제거합니다. 베이스 가중치는 건드리지 않습니다.
  - 두 방식의 결과가 사실상 같다는 것은 작은 모델로 확인했습니다(픽셀 평균 차이 0.28/255).
- 설정: 1024², 8 steps, cfg 1.0, seed 42, 프롬프트 14개(아래 표). 체크포인트 하나당 약 45초 `[추정]`.
- 결과물:
  - `DiffSynth-Studio/models/train/sn15_style_{A,B}/samples/step-N.jpg`: 그 step의 14장을 한 장에 모은 이미지
  - `.../samples/step-N/<id>.png`: 원본 크기 이미지
  - **`step-0`은 화풍 LoRA가 아직 0인 상태**라서 "베이스 + 8-step"(대조군)과 같습니다.
- 관련 인자:
  - `--sample_seed`, `--sample_steps`, `--sample_cfg`, `--sample_size`, `--sample_n_train`
  - `--no_samples`: 샘플 생성 끄기
  - `--no_sample_at_start`: step-0 샘플 생략

### 평가 프롬프트 (14개)
| id | 종류 | 보는 것 |
|---|---|---|
| n01–n08 | 데이터에 없는 새 의상(메이드, 사이버펑크, 기모노, 기사, 치파오, 마법학교, 토끼, 집사). 학습 캡션과 같은 태그 형식 | **화풍 + 프롬프트 충실도** (핵심) |
| p01 | 머리색·눈색 지정 | 캡션에 없던 속성을 제어할 수 있는지 |
| p02 | 짧은 자연어 문장 | 캡션 형식이 아니어도 화풍이 유지되는지 |
| p03 | 아주 짧은 태그 | 묘사가 적어도 화풍이 유지되는지 |
| t01–t03 | 학습 캡션 + GT | 화풍이 GT와 맞는지(참고용). 단, GT를 **그대로 복제**하면 과적합 신호 |

프롬프트 목록은 `validate.py`의 `NOVEL`, `PROBES`에 있습니다. 바꾸면 이후 학습부터 반영됩니다.

## 준비 상태 (이미 완료)
- 가상환경 `/root/venvs/sn15` (시스템 torch 2.8 재사용 + DiffSynth-Studio editable 설치)
- 모델: `DiffSynth-Studio/models/SenseNova/` (MoT 35GB, 8-step V2 LoRA)
- `data`: 데이터셋 폴더(NFD 한글 이름)를 가리키는 ASCII 심볼릭 링크
- `meta/train.csv`(560장 전부), `meta/smoke.csv`(20장). `prepare_metadata.py`로 생성했습니다.
- 작은 무작위 모델로 CPU에서 A/B 전체 흐름(학습, 체크포인트, 샘플, 스케줄러 복원, hook 제거)을 시험해 통과했습니다. **실제 모델·GPU로는 아직 돌려보지 않았습니다.**

## 실행 순서

> GPU는 한 번에 하나만 돌리세요. 모든 작업은 `tmux_start.sh`로 **tmux 세션 안에서** 실행합니다. 그러면 ssh 연결이 끊겨도 계속 돕니다.

### tmux 사용법 요약
| 하고 싶은 것 | 명령 |
|---|---|
| 작업 시작 | `bash /root/source/test_SenseNova-U1.5/sn15_train/tmux_start.sh <job>` (`smoke` / `ab` / `A` / `B` / `test`) |
| 진행 화면 보기 | `tmux attach -t sn15` |
| 화면에서 빠져나오기 (작업은 계속) | `Ctrl-b` 누르고 손 뗀 뒤 `d` |
| 창 전환 (0: 학습, 1: nvidia-smi) | `Ctrl-b` 누르고 `0` 또는 `1` |
| 지난 출력 스크롤 | `Ctrl-b` 누르고 `[` → 방향키/PgUp, `q`로 종료 |
| 작업 중단 | `tmux kill-session -t sn15` |
| 세션 목록 | `tmux ls` |

- 세션 이름은 기본 `sn15`입니다. 이미 있으면 새로 시작하지 않습니다(중복 실행 방지). 학습 프로세스가 이미 돌고 있어도 시작하지 않습니다.
- 작업이 끝나도 창은 열린 채로 남습니다. `job finished, exit code N`으로 성공 여부를 알 수 있습니다. 다음 작업을 시작하기 전에 `tmux kill-session -t sn15`로 닫으세요.
- 모든 출력은 `logs/`에도 저장됩니다. 파일명은 `smoke_A.log`, `train_A.log`, `run_ab.log` 등입니다.
- 서버나 컨테이너 자체가 재시작되면 tmux 세션도 사라지고 학습이 멈춥니다. 그때는 체크포인트에서 `--lora_checkpoint`로 이어서 학습해야 합니다.
- tmux가 처음이면 `tmux_start.sh test`로 먼저 동작을 확인해 보세요(몇 초면 끝남, GPU 사용 안 함).

### 1) 시험 학습 (A, B 합쳐 약 20분)
```bash
bash /root/source/test_SenseNova-U1.5/sn15_train/tmux_start.sh smoke
tmux attach -t sn15
```
`smoke.sh A`, `smoke.sh B`를 차례로 실행합니다. 각각 20장, step 10마다 저장하고, step-0, step-10, step-20에서 샘플을 그립니다.

확인할 것:
- **A**: 로그에 `294 tensors are fused by LoRA.`가 찍히는지. 8-step V2를 합칠 때 나오는 메시지입니다. "Alpha detected" 경고는 정상입니다.
- 다른 터미널에서 `nvidia-smi`로 본 피크가 **약 35–37GiB**인지(샘플링 포함). 42GiB에 가까우면 알려주세요.
- 속도가 **약 1.15–1.4 s/it**인지
- `[samples] step-0: 14 images in …s` 로그가 나오는지. **step-0 샘플이 A와 B에서 거의 같은지**(둘 다 베이스 + 8-step)
- `DiffSynth-Studio/models/train/sn15_smoke_{A,B}/`에 `step-10.safetensors`와 `samples/step-10.jpg`가 생기는지

### 2) 본 학습 A → B (약 7.5시간)
시험 학습 세션을 닫고 시작합니다.
```bash
tmux kill-session -t sn15
bash /root/source/test_SenseNova-U1.5/sn15_train/tmux_start.sh ab
```
- 진행 상황 보기: `tmux attach -t sn15`. 또는 tmux 밖에서 로그만 보려면:
  ```bash
  tail -f /root/source/test_SenseNova-U1.5/sn15_train/logs/train_A.log
  ```
- 학습 중에도 `.../sn15_style_A/samples/step-N.jpg`를 열어 바로 확인할 수 있습니다.
- A나 B 중 하나만 돌리려면 `tmux_start.sh A`(또는 `B`)를 쓰면 됩니다.
- 설정을 바꿔서 단독 실행할 때는 tmux 세션 안에서 직접 실행하세요.
  ```bash
  tmux new -s sn15x
  bash train.sh A --learning_rate 5e-5
  ```
- loss 기록: `.../sn15_style_{A,B}/loss.csv`와 tensorboard_log. timestep을 무작위로 뽑기 때문에 원래 들쭉날쭉합니다. **loss로 체크포인트를 고르지 마세요.**

### 3) A vs B 비교 (run_ab.sh가 마지막에 자동 실행, CPU만 사용)
```bash
/root/venvs/sn15/bin/python /root/source/test_SenseNova-U1.5/sn15_train/compare_samples.py
```
- 결과는 `eval/progress/`에 생깁니다.
  - `<id>.jpg`: 프롬프트별로 행 = A/B, 열 = step-0 … step-10080
  - `all_A.jpg`, `all_B.jpg`: 행 = 프롬프트, 열 = step
- 학습 중간에 실행해도 됩니다(그때까지의 step만 나옴). 열이 너무 많으면 `--every 1000`을 붙이세요.

### 4) (선택) 추가 검증: `validate.py`
학습이 끝난 뒤 후보 체크포인트만 seed를 늘려 다시 보거나, 50-step 참고 이미지를 볼 때 씁니다.
```bash
/root/venvs/sn15/bin/python validate.py --steps 3000,5000 --seeds 42,1234,7
/root/venvs/sn15/bin/python validate.py --steps 5000 --ref50 B:5000
```

### 5) 고르는 기준
화풍이 충분히 들어왔으면서 아래 세 가지를 만족하는 **가장 이른 step**을 고릅니다.
1. step-0(대조군)만큼 선명함이 유지된다. 8-step 증류가 깨지면 흐려지거나 대비가 약해집니다.
2. 새 의상 프롬프트(n01–n08)의 의상·색·소품을 잘 따른다. 학습 데이터의 의상으로 끌려가지 않는다.
3. t01–t03이 학습 이미지를 그대로 복제하지 않는다.

A와 B를 그 기준으로 나란히 비교하세요.

## 사용 (최종 선택 후)
루트의 [`inference.py`](../inference.py)를 쓰세요. `PROMPT`와 `STEP8` 등을 파일 맨 위에서 바꿀 수 있습니다. 내부적으로는 아래와 같이 동작합니다.
```python
pipe.load_lora(pipe.dit, ".../SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors")
pipe.load_lora(pipe.dit, ".../sn15_style_A/step-XXXX.safetensors", alpha=1.0)
image = pipe(prompt="<의상 묘사, 학습 캡션과 같은 태그 형식>", seed=0, height=1024, width=1024,
             num_inference_steps=8, cfg_scale=1.0, shift=3.0)
```

## 파일 목록
| 파일 | 역할 |
|---|---|
| `train.sh` | `train.sh <A\|B> [추가 인자]`. 학습 설정이 모여 있는 곳 |
| `train_sn15.py` | 공식 `train.py` + 체크포인트마다 8-step 샘플 생성 |
| `smoke.sh` | `smoke.sh [A\|B]`. 20장 시험 학습 |
| `tmux_start.sh` | `tmux_start.sh [smoke\|ab\|A\|B\|test]`. tmux 세션 `sn15`에서 작업 실행(ssh가 끊겨도 유지) |
| `run_ab.sh` | A → B 학습 → compare_samples 순차 실행 (화면 출력 + `logs/train_{A,B}.log`) |
| `compare_samples.py` | 학습 중 생성된 샘플로 A/B 진행 grid 생성(CPU) |
| `validate.py` | (선택) 특정 체크포인트를 여러 seed와 50-step 참고 이미지로 추가 검증 |
| `prepare_metadata.py` | png/txt → `meta/train.csv`, `smoke.csv` (`max_timestep_boundary=0.9`) |
