# SenseNova-U1.5 화풍 LoRA 학습 정리 (8-step 기반)

> 작성일: 2026-10-13 · 결과 갱신: 2026-10-14 · 대상 repo: `DiffSynth-Studio` v2.1.9 (로컬) · GPU: A40 46GB × 1
> `[검증]` = 코드/실측으로 확인, `[추정]` = 계산·추론, `[미검증]` = 실제 실행 전까지 모름
> 이 문서는 **"결과 요약"(최종 상태)**과 **0–9장(계획 당시의 분석과 근거)**으로 나뉩니다. 서로 다른 부분이 있으면 결과 요약이 우선입니다.

---

## 결과 요약 (2026-10-14)

### 결론
- **B안을 채택합니다.** 베이스에서 화풍 LoRA를 학습하고, 추론할 때 8-step LoRA를 함께 붙이는 방식입니다.
- **A안(8-step LoRA를 합친 뒤 학습)은 실패했습니다.** 1000 step부터 이미지가 흐릿하고 반투명해졌고, 끝까지 회복되지 않았습니다. 4장에서 미리 짚었던 "8-step 증류 효과가 깨지는" 위험이 그대로 나타났습니다.
- 사용 파일:
  - 학습된 LoRA: `DiffSynth-Studio/models/train/sn15_style_B/step-*.safetensors`
  - 추론 스크립트: [`inference.py`](inference.py) (기본값 `step-10080`)

### A안 vs B안
두 안은 같은 데이터·같은 설정이고, 다른 점은 학습할 때 8-step LoRA를 합치느냐 하나뿐입니다. 샘플은 모두 8 steps / cfg 1.0 / seed 42로 그렸습니다.

| | A안 | B안 |
|---|---|---|
| 학습 | 베이스 + 8-step V2(fused) 위에서 화풍 LoRA 학습 | 베이스 위에서만 화풍 LoRA 학습 |
| 추론 | 베이스 + 8-step + 화풍 LoRA | 같음 |
| step-0 (학습 전) | 선명함 | A와 거의 같음 → 샘플링 코드는 정상 |
| step-1000 이후 | **흐릿하고 반투명함, 끝까지 회복 안 됨** | **선명함, 화풍 적용** |
| 프롬프트 충실도 | 의상 요소는 보이지만 형체가 뭉개짐 | 새 의상 8개를 소품까지 잘 따름(메이드+쟁반, 기모노+부채, 집사+촛대+모노클 등) |
| 캡션에 없던 속성 (p01: 초록 머리, 빨간 눈) | 흐릿하게 반영 | 반영됨 |
| 다른 형식의 프롬프트 (p02 자연어, p03 짧은 태그) | 흐릿함 | 화풍이 유지됨 |
| 과적합 (t01–t03, GT와 비교) | 판단 불가 | 10,080 step까지 GT를 그대로 복제하지 않음(의상만 캡션대로 따르고 머리색·포즈는 매번 다름) |

**A가 실패한 이유** `[추정]`
- 일반 flow-matching 손실은 노이즈 단계마다 "가능한 정답들의 평균"을 예측하도록 학습시킵니다.
- 8-step 모델은 그 평균 대신 몇 step 만에 선명한 한 장을 고르도록 증류돼 있습니다. 그 위에서 이 손실로 학습하면 증류된 동작이 덮어씌워집니다.
- 그 결과 평균에 가까운 흐릿하고 반투명한 이미지가 나옵니다.
- B는 8-step LoRA를 학습에 넣지 않으므로 이 문제가 없습니다.

비교 이미지는 로컬에만 있습니다(`sn15_train/eval/`는 gitignore).
- `sn15_train/eval/ab_summary/ab_novel_1.jpg`, `ab_novel_2.jpg`, `ab_probe_train.jpg`: 프롬프트별 A/B를 step 0–10080에 걸쳐 비교
- `sn15_train/eval/ab_summary/B_vs_GT.jpg`: 학습 캡션에 대한 B 결과와 정답 이미지 비교
- `sn15_train/eval/progress/`: 전체 step 진행 이미지(`compare_samples.py` 출력)

### 실측 수치 `[검증]`
| 항목 | 값 |
|---|---|
| 학습 시간 | 변형당 약 3시간 23분. A: 10-13 11:23 → 14:46, B: 14:46 → 18:10 |
| 학습 속도 | 약 1.11–1.14 s/it |
| 학습 중 샘플 생성 | 14장 기준 A 41초, B 45초. B는 8-step을 hook으로 얹어서 조금 느림 |
| 시험 학습 | A는 `294 tensors are fused by LoRA.` 확인, A/B 모두 오류 없이 체크포인트와 샘플 생성 |
| loss | 처음부터 끝까지 약 0.052–0.059로 거의 평평함. 품질 판단에는 쓸 수 없음 |
| 디스크 | A 4.3GB, B 4.4GB, 시험 학습 출력 약 0.9GB (모두 `DiffSynth-Studio/models/train/`) |
| GPU 메모리 피크 | **기록하지 않음.** 학습이 OOM 없이 끝났으므로 45GiB 이내였던 것만 확실함 |

### 추론 (`inference.py`)
파일 맨 위의 `PROMPT`를 바꾸고 아래 명령으로 실행합니다. 결과는 `outputs/`에 저장됩니다(gitignore).
```bash
/root/venvs/sn15/bin/python /root/source/test_SenseNova-U1.5/inference.py
```

| `STEP8` | 구성 | 설정 | 1장당 시간 `[추정]` |
|---|---|---|---|
| `True` | 베이스 + 8-step V2 + 화풍 LoRA(B) | 8 steps, cfg 1.0 | 약 3초 |
| `False` | 베이스 + 화풍 LoRA(B) | 50 steps, cfg 4.0 | 약 30초 |

- 그 밖의 설정: `STYLE_STEP`(체크포인트, 기본 10080), `STYLE_ALPHA`(화풍 강도), `SEED`, `NUM_IMAGES`, `HEIGHT`/`WIDTH`
- 두 모드 모두 한 번씩 실행해 봤습니다(사용자, seed 42, 같은 프롬프트 1개). 둘 다 학습한 화풍으로 나왔습니다.
  - 50-step: 작은 디테일(재킷의 "K" 패치, 토끼 패치)이 더 선명합니다.
  - 8-step: 조금 부드럽게 나옵니다.
  - 표본이 1장이라 일반화하기는 이릅니다.
- `STEP8=False` 모드에서 주의할 점:
  - LoRA는 σ≥0.25 구간만 학습했습니다(`max_timestep_boundary=0.9`). 50-step의 마지막 세부 묘사 단계는 학습되지 않은 상태로 LoRA가 적용됩니다.
  - cfg 4의 "프롬프트 없는 경로"도 학습하지 않았습니다.
  - 색이 과하면 `CFG_SCALE`을 3.0 정도로 낮춥니다.

### 남은 일
- [ ] **체크포인트 최종 선택**: 지금까지는 seed 42 하나로만 봤습니다. 후보를 seed 여러 개로 다시 봅니다.
  - 학습 초기는 흔들림이 있었고(2000 step에서 p02에 배경이 그려짐) 3000 step 이후로 안정적이었습니다.
  - 명령: `sn15_train/validate.py --runs B=<B 경로> --steps 3000,5000,7000,10080 --seeds 42,1234,7`
- [ ] `STEP8=False`(50-step) 품질을 seed와 프롬프트를 늘려 확인합니다.
- [ ] 필요 없는 출력 정리: A 체크포인트(4.3GB)와 시험 학습 출력(약 0.9GB)은 삭제해도 됩니다. 아직 지우지 않았습니다.
- [ ] (선택) 2048² 등 다른 해상도에서 화풍이 유지되는지 확인합니다(학습은 1024²).

### 진행 기록
| 날짜 | 내용 |
|---|---|
| 10-13 | 데이터·코드 분석(멀티 에이전트 조사 + 적대적 검증) → 8-step 기반 LoRA 학습 방향 결정 |
| 10-13 | 사용자 결정: 트리거 없음, 평가용 데이터 분리 없음(560장 전부 학습), A/B 비교, 10,080 step / 500 step 저장, 학습 중 샘플 생성, tmux 실행 |
| 10-13 | 환경 구축(venv `/root/venvs/sn15`, 모델 다운로드와 SHA256 검증), `sn15_train/` 키트 작성(작은 모델로 CPU 검증), 커밋 `2e79a00` |
| 10-13 | 사용자가 시험 학습(A, B)과 본 학습(A → B, 각 약 3.4시간)을 tmux로 실행 |
| 10-14 | A/B 결과 비교 → **B 채택**. `inference.py` 작성(`STEP8`로 8-step / 50-step 전환) |

---

## 0. 한눈에 보기 (계획 당시)

**확정된 결정 (2026-10-13)**
- 트리거 단어 없음. 원본 캡션을 그대로 사용합니다(이 화풍만 생성할 예정). → 8장 1번
- A안과 B안을 비교 학습합니다(A: 8-step을 합친 뒤 학습, B: 베이스에서만 학습). 둘 다 8-step으로 평가합니다. → **결과: B 채택**
- 학습은 사용자가 직접 실행합니다. 실행 키트는 [`sn15_train/`](sn15_train/README.md)에 있습니다.

| 항목 | 결론 |
|---|---|
| 학습 방식 | 계획: A안(8-step LoRA를 합친 뒤 학습, `--preset_lora_path`) + 대조군 B안 → **결과: B안 채택, A안 실패** (결과 요약 참고) |
| 베이스 모델 | `SenseNova/SenseNova-U1.5-8B-MoT` (정식 릴리스) |
| 8-step LoRA | `SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors` (공식 권장 V2) |
| 추론 | 베이스 + 8-step LoRA + 화풍 LoRA, **8 steps / cfg 1.0 / shift 3.0 / 1024²** |
| 학습 도구 | `sn15_train/train_sn15.py` = 공식 `train.py`(수정 없이 불러 씀) + 체크포인트마다 8-step 샘플 생성 |
| A40 1장으로 가능? | **가능.** 비슷한 구조로 측정한 피크 VRAM은 약 34.7GiB, step당 약 1.13초 `[검증: 동일 구조 replica 실측]` |
| 학습 길이 | **10,080 step**(560장 × 18 epoch), 500 step마다 체크포인트와 샘플 생성. 변형당 약 3.5–4시간 `[추정]` |
| 가장 큰 위험 | 일반 flow-matching 손실로 학습하면 **8-step 증류 효과(선명도)가 약해질 수 있음**. 체크포인트마다 8-step으로 평가하고, 대조 실험(B안)을 함께 돌릴 것 → **실제로 A안에서 발생** |

---

## 1. 환경 현황

| 항목 | 값 | 비고 |
|---|---|---|
| GPU | NVIDIA A40, 46,068 MiB (≈45 GiB) | Ampere: bf16은 되지만 fp8 tensor core 없음 |
| **RAM** | **cgroup 제한 약 50GB (46.6 GiB)** | `free`의 503GB는 호스트 값이라 실제로는 못 씀 |
| **CPU** | **cgroup quota 7.65코어** | `nproc`의 96은 호스트 값 |
| 디스크 | /root 여유 약 198GB | 모델 35GB + LoRA 1.6GB + 체크포인트(개당 약 200MB) |
| Python / torch | 3.12 / 2.8.0+cu128 | 공식 upstream도 torch 2.8 |
| DiffSynth | venv `/root/venvs/sn15`에 설치함 | 시스템 pip가 PEP 668로 막혀 있어 `--system-site-packages` venv 사용(torch 2.8 재사용). transformers 5.19, peft 0.21, accelerate 1.15 |
| 모델 가중치 | `DiffSynth-Studio/models/SenseNova/`에 다운로드 | MoT 35GB + 8-step V2 LoRA 815MB |

RAM 제한 때문에 **CPU offload 계열 옵션(`--enable_model_cpu_offload`, `--initialize_model_on_cpu`)은 쓰지 말 것.** 가중치 33GB가 호스트 RAM을 거치면서 OOM-kill(traceback 없이 `Killed`만 출력)될 위험이 큽니다.

---

## 2. 데이터셋 분석

경로: `dataset/0+1등급_1024_gemini3.8flash/`

| 항목 | 값 |
|---|---|
| 구성 | PNG 560개 + TXT 560개, 같은 파일명끼리 1:1로 짝지어짐 (누락 0) |
| 해상도 | 전부 1024×1024 → 학습 시 리사이즈 없음, 이미지 토큰 1024개 |
| 모드 | RGBA. 알파 최소값은 모든 이미지에서 191(글로우·외곽선의 반투명 부분). **RGB가 이미 회색 배경과 합성돼 있음** |
| 그림 | SD(치비) 비율의 전신 아바타 캐릭터, 단색 회색 배경, 일관된 게임 화풍 |
| 캡션 | 한 줄짜리 영어 태그·구문, 38–82단어. **의상만 묘사함** |
| 그룹 | `1120_GACHA-318_1/_2`처럼 같은 세트의 다른 의상 변형이 있음. 실제 세트 수는 413개 |

### 꼭 알아둘 점
1. **폴더명이 NFD 한글**(macOS zip)이라 직접 타이핑(NFC)하면 경로를 못 찾습니다. ASCII 심볼릭 링크를 만들어 쓰는 게 안전합니다.
   ```bash
   ln -s "$(ls -d /root/source/test_SenseNova-U1.5/dataset/*/ | head -1)" /root/source/test_SenseNova-U1.5/data_chibi
   ```
2. **RGBA는 그대로 둬도 됩니다.** 로더가 `convert("RGB")`로 알파만 버리는데, RGB가 이미 합성된 결과라 정답입니다. 회색 배경에 다시 합성하면 반투명 부분이 이중으로 섞여 탁해지니 **추가 전처리는 하지 마세요.**
3. **캡션에 없는 속성의 처리**
   - 항상 똑같은 속성(화풍, SD 비율, 회색 배경, 전신 구도)은 LoRA에 **기본값으로 굳어집니다.** 화풍 학습이 목적이니 바람직합니다.
   - 이미지마다 다르지만 캡션에 없는 속성(머리색, 눈색, **성별**, 포즈)은 의상 단어와 **얽혀서** 학습됩니다. 예: `prince`(30회), `ouji`(16회)가 남성과 엮임. 그래서 "boy + 여성 의상" 같은 조합이 잘 안 될 수 있습니다.
4. **세트 구조** (참고): 같은 세트의 `_1/_2` 변형이 있고, 앞 번호가 다른 같은 세트도 있습니다(예: `1941_SUGOROKU-1-MUSHIBA_2`와 `1942_SUGOROKU-1-MUSHIBA_1`). 실제 세트 수는 413개입니다. 현재는 평가용 데이터를 떼지 않으므로 영향이 없고, 나중에 데이터를 나눌 일이 생기면 세트 단위로 나눠야 합니다.
   ```python
   group = re.sub(r"^\d+_", "", re.sub(r"_(\d+(-\d+)?|F\d+)$", "", stem))
   ```

---

## 3. 모델 구조 핵심 (학습에 영향을 주는 것만)

- **총 17.5B 파라미터** = 이해(understanding) 브랜치 약 9.4B + 생성(generation) 브랜치 약 8.2B. bf16으로 32.7GiB. 이름의 "8B"는 브랜치 하나 크기입니다.
- **VAE가 없는 픽셀 공간 모델**입니다. 32px 패치 하나가 토큰 하나라서 1024²는 1024 토큰입니다. 별도 텍스트 인코더도 없습니다.
- **텍스트는 이해 브랜치(동결)만 통과**해 KV prefix가 되고, 이미지 토큰은 생성 브랜치(`*_mot_gen`)를 통과하며 그 prefix를 참조합니다.
  - 따라서 LoRA 대상은 **생성 브랜치 7종 × 42층 = 294개 모듈**입니다.
    `q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj`
  - LoRA는 텍스트 표현을 바꾸지 못합니다. 새 단어의 "의미"를 가르칠 수는 없고, 기존 토큰에 반응하는 방식만 바뀝니다.
- 프롬프트 템플릿: ChatML 형식에 시스템 프롬프트(약 240토큰)가 붙고 `<img>`로 끝납니다. **길이 제한·잘림은 없습니다.**
- 학습 손실(`FlowMatchSFTLoss`): timestep을 1000단계, shift 3.0 스케줄에서 균등하게 뽑아 velocity MSE를 계산합니다. 노이즈 스케일은 해상도에 따라 다릅니다(1024²에서 4.0). 학습과 추론의 shift가 3.0으로 일치합니다.
- 학습 중에는 cfg_scale=1(조건 있는 쪽만)로 돌고, caption dropout은 없습니다. 8-step 추론도 cfg 1이라 이 점은 **오히려 잘 맞습니다.**

### 모델·LoRA 종류

| 이름 | 정체 | 용도 |
|---|---|---|
| `SenseNova-U1.5-8B-MoT` | 정식 릴리스(SFT 후 RL과 MOPD까지 거침) | **베이스로 사용** |
| `SenseNova-U1.5-8B-MoT-SFT` | SFT까지만 한 체크포인트 | 대안 베이스(A/B 실험용) |
| `-Edit`, `-SFT-Edit` | 별도 가중치가 아님. 같은 모델의 편집 모드 | T2I에는 무관 |
| `SenseNova-U1.5-8B-MoT-LoRAs` | **8-step 증류 LoRA**. rank 128, 같은 294개 모듈, 파일당 약 815MB | 8 steps / cfg 1.0 / shift 3.0 |
| └ `...-LoRA-8step-V2.safetensors` | 최신·권장(색 균형 개선, 과샤프닝 감소) | **이것을 사용** |
| └ `...-LoRA-8step.safetensors` | V1. 로컬 예제 코드는 이 파일을 참조 | |

8-step 추론 시 실제로 쓰는 σ 값은 `1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3`이고, **σ < 0.3 구간은 전혀 쓰지 않습니다.** `[검증: 스케줄 계산]`

---

## 4. 8-step 위에서 LoRA를 학습하는 방법과 위험

### A안 (선택한 방향): 8-step LoRA를 합친 뒤 화풍 LoRA 학습
```
학습:  W = W_base + ΔW_8step(합침, 고정)  →  그 위에 화풍 LoRA(학습 대상)
추론:  W_base + ΔW_8step + ΔW_style        →  8 steps, cfg 1.0
```
- `train.py`가 이미 지원합니다: `--preset_lora_path <8-step 파일 경로> --preset_lora_model dit`
  (`training_module.py:363-364`에서 `pipe.load_lora(dit, path)`로 합친 뒤 새 LoRA를 주입) `[검증: 코드]`
- `--preset_lora_path`에는 **로컬 파일 경로**(str)를 줘야 합니다. 모델 ID 형식은 안 되므로 먼저 다운로드해 두세요.
- 저장되는 체크포인트에는 **화풍 LoRA만** 들어갑니다(학습되는 파라미터만 저장). 그래서 추론할 때는 **학습에 쓴 것과 같은 버전의 8-step LoRA**(V2)를 반드시 먼저 로드해야 합니다.

**위험 요소**
- 학습 손실은 일반 flow-matching(무작위 timestep에서 평균적인 정답을 예측)입니다. 증류된 모델은 몇 step 만에 선명한 결과를 내도록 조정돼 있는데, 이 손실은 그 동작을 **다시 다단계 모델 쪽으로 끌어당깁니다.** 학습이 길어질수록 8-step 출력이 흐려지거나 대비가 약해질 수 있습니다. `[추정]`
- 커뮤니티 사례가 없습니다. SenseNova-U1.5 화풍 LoRA 공개 결과를 찾지 못했습니다. `[미검증]`

**완화책**
1. **timestep 범위를 8-step 구간으로 제한**합니다. 메타데이터에 `max_timestep_boundary` 열을 넣고 `--extra_inputs max_timestep_boundary`로 전달합니다. `loss.py:11`이 이 값을 읽습니다. `[검증: 코드]`

   | 값 | 학습되는 최소 σ | 비고 |
   |---|---|---|
   | 0.875 | 0.30 | 8-step 마지막 σ와 같음 |
   | **0.9** | **0.25** | **권장(약간의 여유)** |
   | 0.98 | 0.058 | 50-step 추론용 |

   추가 효과: σ가 아주 작은 구간은 1/σ로 증폭된 손실 스파이크를 만드는데, 이 구간이 빠져 학습도 안정됩니다.
2. **체크포인트를 자주 저장**하고(`--save_steps 500`) **저장할 때마다 학습 프로세스 안에서 8-step 샘플을 바로 그립니다**(`train_sn15.py`). 8-step은 1024² 한 장에 몇 초면 되므로 부담이 적습니다.
3. LR은 1e-4에서 시작하고, 초반부터 선명도가 떨어지면 5e-5로 낮춥니다.

### B안 (대조 실험 권장): 베이스에서만 학습하고 추론 때 합치기
- `--preset_lora_path` 없이 베이스 MoT에서 화풍 LoRA를 학습합니다. 추론할 때 8-step V2와 화풍 LoRA를 둘 다 로드합니다.
- 다른 모델(Qwen-Image Lightning 등)에서는 흔히 쓰는 방식이지만, 두 LoRA가 같은 294개 행렬을 건드리므로 잘 맞물린다는 보장은 없습니다. `[미검증]`
- 장점: 증류를 망가뜨리지 않고, 50-step 고품질 추론용으로도 쓸 수 있습니다.
- **같은 데이터와 설정으로 A·B를 각각 약 2시간씩 돌려 8-step 결과를 나란히 비교하는 것을 권장합니다.**

### C안 (참고만): direct distill
- DiffSynth에 `DirectDistillLoss`가 있지만 SenseNova `train.py`에는 연결돼 있지 않습니다(Qwen-Image 전용).
- 실제 데이터가 아니라 "교사 모델이 생성한 이미지"가 필요합니다(예: B안 LoRA로 50-step 생성 → 그 결과로 8-step 학습).
- A·B 모두 실패할 때만 고려합니다.

---

## 5. 단계별 진행 절차

모든 명령은 `/root/source/test_SenseNova-U1.5/DiffSynth-Studio`에서 실행합니다(스크립트 경로가 repo 기준 상대경로).

> **Step 1–3은 완료했습니다.** 실제 실행은 [`sn15_train/README.md`](sn15_train/README.md)를 따르세요. 아래는 그 과정의 기록입니다.

### Step 1. 설치 (완료)
```bash
python3 -m venv --system-site-packages /root/venvs/sn15     # 시스템 pip는 PEP 668로 차단됨
cd /root/source/test_SenseNova-U1.5/DiffSynth-Studio
/root/venvs/sn15/bin/python -m pip install -e . tensorboard
```
- `[quant]`, `[all]` extra는 설치하지 마세요(entropack이 torch 2.10 이상을 요구함).
- 전체 학습용 DeepSpeed(`[training]`)는 LoRA에는 필요 없습니다.

### Step 2. 모델 다운로드 (약 36GB, 완료)
```bash
/root/venvs/sn15/bin/modelscope download --model SenseNova/SenseNova-U1.5-8B-MoT \
  --local_dir ./models/SenseNova/SenseNova-U1.5-8B-MoT
/root/venvs/sn15/bin/modelscope download --model SenseNova/SenseNova-U1.5-8B-MoT-LoRAs SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors \
  --local_dir ./models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs
export DIFFSYNTH_SKIP_DOWNLOAD=True   # 이후 매 실행 때 원격 조회를 하지 않음
```
- 기본 소스는 ModelScope입니다. HF를 쓰려면 `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`를 설정하세요.
- `./models`는 repo의 `.gitignore`에 포함돼 있습니다.

### Step 3. 데이터 준비 (완료: `sn15_train/prepare_metadata.py`)
- 결과: 학습 560장 전부, 스모크 테스트 20장. 트리거 없음, `max_timestep_boundary=0.9`.
- 심볼릭 링크는 `sn15_train/data`로 만들었습니다(아래의 `data_chibi`는 처음 계획에서 쓴 이름).

1. ASCII 심볼릭 링크 `data_chibi`를 만듭니다(2장 참고).
2. **메타데이터 CSV**를 만듭니다. 형식은 `image,prompt[,max_timestep_boundary]`입니다.
   - 캡션에 쉼표가 많으므로 반드시 `csv` 모듈의 `QUOTE_ALL`로 작성합니다(문자열을 직접 이어 붙이지 말 것).
   - 컬럼 이름은 **정확히 `prompt`**여야 합니다(`train.py:57`).
   - 메타데이터 없이 폴더를 지정해도 txt를 자동으로 읽지 않습니다. `.pth` 캐시 모드로 빠져 실패합니다.
   - `image`는 `--dataset_base_path` 기준 상대경로(파일명)입니다.
   ```csv
   "image","prompt","max_timestep_boundary"
   "1113_GACHA-75_1.png","<트리거>, festive santa mini dress with ...","0.9"
   ```
3. ~~홀드아웃~~: 처음에는 세트 단위로 떼어 둘 계획이었지만 **하지 않기로 했습니다**(Step 6 참고).
4. 메타데이터는 데이터셋 폴더 밖(예: `/root/source/test_SenseNova-U1.5/train_meta/`)에 두면 원본이 깨끗하게 유지됩니다.

### Step 4. 스모크 테스트 (약 10분)
- 아래 학습 명령에서 `--num_epochs 1 --save_steps 20`으로 바꾸고, 20행 정도의 작은 CSV로 실행합니다.
- 확인할 것:
  - `nvidia-smi` 피크가 **약 34.5–36GiB**인지 (37GiB를 넘으면 다른 프로세스를 의심)
  - tqdm 속도가 **약 1.15–1.35 s/it**인지
  - 8-step LoRA 로드 로그 `N tensors are fused by LoRA.`의 N이 **294**인지 `[추정: 7모듈×42층]` (경로나 키가 틀려도 경고 없이 0개만 합쳐질 수 있음)
  - `step-20.safetensors`가 생성되는지

### Step 5. 본 학습 (A안)
> 실제로는 `sn15_train/train.sh A` 또는 `B`로 실행합니다(경로는 `sn15_train/` 기준으로 정리됨). 아래는 A안 명령의 원형입니다.
> 비교를 공정하게 하려고 B안도 같은 `max_timestep_boundary=0.9`를 씁니다. 8-step(+stack) 추론도 σ≥0.3 구간만 쓰기 때문입니다. 두 안의 차이는 preset LoRA 여부 하나뿐입니다.

```bash
cd /root/source/test_SenseNova-U1.5/DiffSynth-Studio
export DIFFSYNTH_SKIP_DOWNLOAD=True
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision no --dynamo_backend no \
  examples/sensenova_u1/model_training/train.py \
  --dataset_base_path /root/source/test_SenseNova-U1.5/data_chibi \
  --dataset_metadata_path /root/source/test_SenseNova-U1.5/train_meta/metadata_train.csv \
  --data_file_keys image \
  --extra_inputs max_timestep_boundary \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --dataset_num_workers 2 \
  --model_id_with_origin_paths "SenseNova/SenseNova-U1.5-8B-MoT:model*.safetensors" \
  --tokenizer_config "SenseNova/SenseNova-U1.5-8B-MoT:./" \
  --preset_lora_path ./models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors \
  --preset_lora_model dit \
  --lora_base_model dit \
  --lora_target_modules "q_proj_mot_gen,k_proj_mot_gen,v_proj_mot_gen,o_proj_mot_gen,mlp_mot_gen.gate_proj,mlp_mot_gen.up_proj,mlp_mot_gen.down_proj" \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --num_epochs 18 \
  --save_steps 500 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path ./models/train/chibi_style_8step_lora \
  --use_gradient_checkpointing \
  --enable_csv_log
```
- **B안**은 `--preset_lora_path`와 `--preset_lora_model` 두 줄을 빼고, `--output_path`만 바꿔서 실행하면 됩니다.
- 출력은 `step-500.safetensors`, `step-1000.safetensors`, … `step-10080.safetensors` 순서로 생기고, 하나당 약 200MB입니다. 샘플은 `samples/step-N.jpg`에 생깁니다.

### Step 6. 평가
- 체크포인트마다 **베이스 + 8-step V2 + 화풍 LoRA**, 8 steps / cfg 1.0 / shift 3.0 / 1024² / 고정 seed 2개 이상으로 그리드를 만듭니다.
- **대조군**: 베이스 + 8-step V2만 쓴 결과(화풍 없음). 항상 같이 렌더링합니다.
- 프롬프트 구성 (`sn15_train/validate.py`, 14개):
  - 데이터에 없는 새 의상 8개(학습 캡션과 같은 태그 형식): 화풍 + 프롬프트 충실도. **핵심 지표**입니다.
  - 형식이 다른 프롬프트 3개(머리색·눈색 지정, 짧은 자연어 문장, 아주 짧은 태그)
  - 학습 캡션 3개 + 정답 이미지: 화풍 비교용. 그대로 복제하면 과적합 신호입니다.
- 평가용 데이터는 따로 떼지 않습니다(560장 전부 학습). 목표가 정답 재현이 아니라 화풍과 프롬프트 충실도라서, 새로 쓴 프롬프트로 일반화를 확인하는 것으로 충분합니다.
- 체크포인트를 바꿀 때는 LoRA를 hot-load로 올렸다 내립니다(`pipe.enable_lora_hot_loading(pipe.dit)` 후 `load_lora`, 끝나면 `pipe.clear_lora()`). fused 로드는 되돌릴 수 없습니다.
- **고르는 기준**: 화풍이 충분히 들어왔으면서 ① 8-step 선명도가 대조군만큼 유지되고 ② 새 의상 프롬프트를 따르고 ③ 학습 이미지를 그대로 복제하지 않는 **가장 이른 체크포인트**를 고릅니다.
- **loss 곡선으로는 고르지 마세요.** timestep을 무작위로 뽑기 때문에 원래 들쭉날쭉합니다.

### Step 7. 추론 (사용 시)
```python
import torch
from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device="cuda",
    model_configs=[ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors")],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
)
pipe.load_lora(pipe.dit, "./models/SenseNova/SenseNova-U1.5-8B-MoT-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors")
pipe.load_lora(pipe.dit, "./models/train/chibi_style_8step_lora/step-XXXX.safetensors", alpha=1.0)  # alpha로 강도 조절
image = pipe(prompt="<트리거>, ...의상 묘사...", seed=42, height=1024, width=1024,
             num_inference_steps=8, cfg_scale=1.0, shift=3.0)
```
- 1024²에서 학습했으므로 1024²부터 확인합니다. 2048²(기본값)에서 어떻게 일반화되는지는 따로 확인해야 합니다. `[미검증]`
- `load_lora`는 기본적으로 가중치에 영구로 합칩니다(fuse). 다른 LoRA로 바꾸려면 파이프라인을 다시 로드하거나 hot-load를 쓰세요.

---

## 6. 권장 하이퍼파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| LoRA rank | 32 (alpha = rank로 고정, scale 1) | 공식 예제. 16/64는 나중에 비교 |
| LR | 1e-4 (불안정하면 5e-5) | 공식 예제·커뮤니티 기준 |
| 배치 | 1로 고정 (CLI에 옵션 없음) | 늘리려면 `--gradient_accumulation_steps` |
| `--dataset_repeat` | **1** | 공식 예제의 50은 작은 예제 데이터 기준. 560장에 그대로 쓰면 14만 step |
| epoch / step | 18 epoch = 10,080 step | 길게 학습하고 500 step 단위 샘플로 최적 지점을 고릅니다. 커뮤니티 기준(2,000–4,000 step)보다 길어서 앞쪽 step이 최적일 가능성이 큼 |
| 저장 간격 | `--save_steps 500` (micro-step 기준) | 설정하면 epoch 단위 저장은 꺼짐. 저장할 때마다 샘플 생성 |
| timestep 상한 | `max_timestep_boundary = 0.9` | 8-step σ 구간(≥0.25)에 맞춤 |
| 해상도 | 1024² 그대로 (`--max_pixels 1048576`) | 2048²로 업스케일해서 학습하면 흐림을 학습함 |
| gradient checkpointing | 켬 | 끄면 27% 빠르지만 피크 약 42GiB로 여유가 거의 없음 |
| 옵티마이저 / 스케줄러 | AdamW(wd 0.01) / ConstantLR (처음 5 step은 lr/3) | 고정. warmup·decay·clip 없음 |

---

## 7. 함정 체크리스트

- [ ] `--lora_target_modules`를 **반드시 7개 `_mot_gen` 이름으로 명시**합니다.
  - 기본값(`q,k,v,o,ffn.0,ffn.2`)은 아무 모듈에도 매칭되지 않습니다.
  - `gate_proj`처럼 접미사만 쓰면 이해 브랜치까지 건드립니다.
  - 하나만 쓰면 정규식 fullmatch로 처리돼 실패합니다.
- [ ] `--use_gradient_checkpointing`을 명시합니다(파서 기본값이 False).
- [ ] 메타데이터 컬럼명은 `image`, `prompt`. CSV는 QUOTE_ALL로 작성합니다.
- [ ] 데이터셋 경로는 심볼릭 링크 사용(NFD 한글 폴더명 때문).
- [ ] `--preset_lora_path`는 로컬 파일 경로. 로드 메시지에서 **합쳐진 개수가 0이 아닌지** 확인합니다.
- [ ] 추론 때 **학습에 쓴 것과 같은 8-step LoRA 버전(V2)**을 사용합니다.
- [ ] 이어서 학습할 때는 `--lora_checkpoint`를 씁니다. `--resume_from_checkpoint`는 LoRA에서 에러가 납니다. 옵티마이저 상태는 저장되지 않습니다.
- [ ] `full/accelerate_config_zero2.yaml`은 쓰지 마세요(8 GPU + DeepSpeed 전제).
- [ ] CPU offload 계열 옵션은 쓰지 마세요(컨테이너 RAM이 약 50GB로 제한됨).
- [ ] `--fp8_models`를 `--initialize_model_on_cpu`나 `--enable_model_cpu_offload`와 함께 쓰면 device 불일치 에러가 납니다. 애초에 VRAM이 충분하니 fp8도 쓸 필요가 없습니다.
- [ ] 학습 중에는 같은 GPU에서 추론(약 33GiB)을 동시에 돌릴 수 없습니다.
- [ ] `sft:data_process` 2단계 캐싱은 SenseNova에서는 이득이 거의 없습니다(prefix 계산이 dit 안에 있기 때문). 쓰지 마세요.
- [ ] 학습 seed 옵션이 없어 실행마다 결과가 조금씩 다릅니다. A/B 비교 시 이 점을 감안하세요.

---

## 8. 결정 사항

1. **캡션 전략 → 트리거 없이 원본 캡션 그대로 (확정)**
   - 이유:
     - 이 LoRA는 이 화풍 전용이라 스위치가 필요 없습니다.
     - 모든 행에 트리거를 붙여도 트리거는 스위치가 되지 못합니다. 생성 브랜치 LoRA는 프롬프트와 관계없이 모든 이미지 토큰에 적용되기 때문입니다. **화풍의 on/off는 LoRA 로드 여부와 `alpha`로 조절**합니다.
     - 50장 이상의 화풍 데이터는 트리거 없이도 잘 학습된다는 사례가 있습니다.
   - 대신 지킬 것:
     - 추론 프롬프트를 **학습 캡션과 같은 형식**(의상 → 색 → 무늬 → 소재 → 소품 → 테마 → 분위기 순서의 쉼표 태그)으로 씁니다.
     - 짧은 자연어 프롬프트에서도 화풍이 유지되는지는 평가 프롬프트 `p02`로 확인합니다.
   - 남는 한계: 머리색·눈색·성별은 캡션에 없어서 프롬프트로 정확히 제어되지 않을 수 있습니다(`p01`로 확인). 필요하면 나중에 성별 태그나 VLM 리캡션을 추가해 재학습합니다.
   - `chibi`, `anime style` 같은 화풍 단어는 넣지 않습니다(스타일이 그 단어들로 분산됨).
2. **A안·B안 둘 다 학습해 비교 (확정)**: `sn15_train/run_ab.sh`로 순차 실행합니다(합계 약 7.5시간, 학습 중 샘플 생성 포함).
3. **SFT 베이스 비교 여부(보류)**: MoT 결과가 과포화되거나 디테일이 과하면 SFT로 짧게 비교해 볼 수 있습니다.
   - 참고로 DiffSynth PR #1658에는 "8-step LoRA는 SFT에서 학습됐다"는 언급이 있고, HF 모델 카드는 MoT용이라고 해서 서로 엇갈립니다.
4. **데이터 권리**: 모델은 Apache-2.0이지만, 학습 데이터(자사 게임 아트)의 권리 문제는 별개로 확인이 필요합니다.

### 추가로 고려할 수 있는 개선 (지금은 적용하지 않음)
지금 키트는 공식 `train.py`의 학습 코드를 수정 없이 씁니다(`train_sn15.py`는 샘플 생성만 추가). 1차 결과를 본 뒤 필요하면 적용하세요. 아래 항목은 `train.py`를 복사해서 몇 줄 고치는 방식으로 적용할 수 있습니다.
- **LoRA 파라미터를 fp32로 유지**
  - 공식 코드는 LoRA와 AdamW 상태를 bf16으로 둡니다. lr 1e-4에서도 업데이트의 약 1/3이 반올림으로 사라지고, weight decay는 사실상 0이 됩니다(시뮬레이션 결과).
  - 비용: VRAM 약 +0.8GiB, 체크포인트 약 390MB.
- **학습 seed 고정**: 공식 코드에는 seed 옵션이 없어 A/B 차이에 실행 간 노이즈가 섞입니다.
- **uncond dropout(10%)**
  - 공식 upstream 학습은 이렇게 합니다. `is_positive=False`로 주면 추론 때의 unconditional prefix와 정확히 같아집니다.
  - **cfg 1인 8-step 추론에는 영향이 없고**, B안을 50-step cfg 4로 쓸 때만 의미가 있습니다.
- **결정적 validation loss**: 고정 σ와 고정 노이즈로 MSE를 계산해 그리드 외의 객관적 지표를 만듭니다. 단, 과적합을 보려면 학습에 쓰지 않은 이미지가 필요합니다. 그래서 평가용 데이터를 떼지 않기로 한 지금 방침과는 맞지 않습니다.

---

## 9. 참고 위치

| 내용 | 위치 |
|---|---|
| 학습 스크립트 | `DiffSynth-Studio/examples/sensenova_u1/model_training/train.py` (공식). 실행은 `sn15_train/train_sn15.py`로 |
| 공식 LoRA 예제 | `.../model_training/lora/SenseNova-U1.5-8B-MoT.sh` |
| 8-step 추론 예제 | `.../model_inference/SenseNova-U1.5-8B-MoT-LoRA-8step.py` (V1 파일을 참조함) |
| preset LoRA 합치기 | `diffsynth/diffusion/training_module.py:363-364` |
| 손실 / timestep 경계 | `diffsynth/diffusion/loss.py:5-33` |
| CLI 플래그 전체 | `diffsynth/diffusion/parsers.py` |
| 데이터 로더 | `diffsynth/core/data/unified_dataset.py`, `operators.py` |
| 파이프라인·프롬프트 템플릿 | `diffsynth/pipelines/sensenova_u1_image.py`, `diffsynth/models/sensenova_u1_common.py` |
| 모델 문서 | `DiffSynth-Studio/docs/en/Model_Details/SenseNova-U1.md` |
| 8-step LoRA 모델 카드 | https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs |
| 공식 repo | https://github.com/OpenSenseNova/SenseNova-U1 |
| DiffSynth 지원 PR | https://github.com/modelscope/DiffSynth-Studio/pull/1658 |
