# Release Package Manifest — GenAI LSTM Final Submission

**Project Title:** LSTM Next-Word Prediction & Text Generation System  
**Primary Archive File:** `release/GenAI_LSTM_Final_Submission.zip`  
**Total Packaged Files:** 102 files  
**Total Uncompressed Size:** 409.99 MB (429,910,758 bytes)  
**Archive Size (Compressed):** 378.74 MB (397,134,011 bytes)  
**Package SHA-256:** `4d04f647f4e5ccb74fca14b97b3b1363671edad7f43752e135c0476998152b3e`  
**Standalone Run-Readiness:** **YES — Out-of-the-Box Runnable for Inference & Evaluation**  

---

## 1. Model Checkpoint Status & Checksums

| Artifact | Relative Path | Epoch | File Size | Status | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **best_val.pt** | `backend/model_artifacts/checkpoints/best_val.pt` | 46 | 306.1 MB (320,991,519 bytes) | PACKAGED (Primary Inference) | `2ad9f4b1fac120846a652406b45357515486838d85d7450ee91d96262b23185d` |
| **multidomain_adapted.pt** | `experiments/multidomain_v1/checkpoints/multidomain_adapted.pt` | 5 | 102.0 MB (106,996,080 bytes) | PACKAGED (Domain Adaptation Inference) | `a737ed30c48481dd78bf23e52a2ceb5eddeabb33508b6a0ea408afb82f21c875` |
| **latest.pt** | `backend/model_artifacts/checkpoints/latest.pt` | 50 | 306.1 MB (320,989,769 bytes) | EXTERNAL (Training Continuity Only) | `c29b92f0b7e2380df6a74efbde89ea7ad77ff64b7af440773f33a22af907629d` |

### Checkpoint Inclusion & Run-Readiness Policy
- **`best_val.pt` (Epoch 46):** **INCLUDED inside the ZIP archive.** This is the primary trained baseline model loaded by the FastAPI inference service (`backend/app/main.py`) for next-word prediction and greedy/temperature/top-k text generation.
- **`multidomain_adapted.pt` (Epoch 5):** **INCLUDED inside the ZIP archive.** This is the adapted model checkpoint used for multi-domain zero-shot and adapted comparison benchmarks.
- **`latest.pt` (Epoch 50):** **Maintained separately in the project workspace.** This checkpoint represents the final training epoch (step 50) and is required only if continuing LSTM pretraining on WikiText-2. It is omitted from the submission archive to keep the package size well within the 512 MB submission ceiling (~379 MB total).

---

## 2. Quickstart Execution Guide (Post-Extraction)

Because `best_val.pt` and `multidomain_adapted.pt` are bundled inside the package, no external weight downloads or manual file placements are needed to run inference.

### Step 1: Verify Extracted Checkpoints
```bash
python scripts/verify_checkpoints.py
```

### Step 2: Start the FastAPI Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Step 3: Start the Next.js Frontend Dashboard
```bash
cd ../frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with Next-Word Prediction, Text Generation, and the Generalization Benchmark Dashboard.

---

## 3. Packaged Files Inventory & SHA-256 Checksums

The submission ZIP contains all clean source code, configurations, evaluation data, and required model weights. The external `release/` directory is excluded from packaging to prevent circular self-referencing.

| Relative File Path | Size (Bytes) | SHA-256 Checksum |
| :--- | :---: | :--- |
| `.gitignore` | 1,176 | `d151654190e31c70b50ff29995a8dd2c7cfb8d34781bf784cbeb5641245e7bd1` |
| `architecture.md` | 3,631 | `83a28f7a72a729bb99b26d4b2616bc6aeefcb3ebe28cb71d896b67b187ec3293` |
| `backend/__init__.py` | 29 | `ac4124b232f0a58d0bdbc19c584e010e9f29683eca6dcdc6053003f9e28d113f` |
| `backend/app/__init__.py` | 35 | `45390889d8a4a81a685fa8f8fb38422e5f6ff8840d7df1405f088ab5347ae8e2` |
| `backend/app/main.py` | 16,533 | `0001f44f54f2812d8f258648375f07e33c8502c415bde363bec5af9a267e22eb` |
| `backend/app/schemas.py` | 2,171 | `28a7299eea13bafe844e8f213d1edb5702b690766843daa1eeb21d7b43ccaa56` |
| `backend/datasets/processed/.gitkeep` | 52 | `b30d90044dae297b8bb3af0f305c930c16bbad58117f16e6d30fc0ef131f5118` |
| `backend/datasets/processed/metadata.json` | 1,449 | `d5786a6c00944fc264150085065ead4c1838871448f91c49d4fbeaf3cd2cee20` |
| `backend/datasets/raw/.gitkeep` | 34 | `53221ebff3a01b2288b7bcdcfe018ff412afce7c0c1788308b0c82fdd5382f33` |
| `backend/datasets/raw/sample_corpus.txt` | 3,258 | `7601f8e6a08a6148dd54bd7b77f73a3ef082c655864404114dbad6c392581f46` |
| `backend/model/__init__.py` | 49 | `1ca8f535dd442a9034dfbcef8fd0d81f1e57264601545fb8188a181ed482bfc7` |
| `backend/model/config.py` | 1,573 | `971c21501d42296ec424c786a70ab41b18fc81973188b8aa7263f8506d9dac02` |
| `backend/model/lstm.py` | 5,078 | `dae8bf419cca9d2a91529efb3b70ad835f7876d573acf62892267eb5bb3b2be4` |
| `backend/model/tokenizer.py` | 5,165 | `d9e6f00428ce13f4ca59d79e14e50659803cc26e4963bff0a28e6c8a3f2d35c9` |
| `backend/model_artifacts/checkpoints/.gitkeep` | 66 | `129d45584381ed9e5a80cb50fdc18428c22959984f55a9bd28d8748dcd70a9ef` |
| `backend/model_artifacts/checkpoints/best_val.pt` | 320,991,519 | `2ad9f4b1fac120846a652406b45357515486838d85d7450ee91d96262b23185d` |
| `backend/model_artifacts/config/.gitkeep` | 67 | `5519f82c6f5096912bee91bfad1375dc22ab90c4870a5b93cf8f8e24bfe3ad20` |
| `backend/model_artifacts/config/kaggle_epoch_metrics.json` | 2,728 | `a0dc313c6ceda71bce9b377af73d54c8b34568d30c0afa3348b816f15e353be6` |
| `backend/model_artifacts/config/kaggle_training_summary.json` | 221 | `9d87ebbae92a4012ad2e11bf7eec3fe0457f92896ea194235d7662ff4d1bf6d9` |
| `backend/model_artifacts/results/generalization_benchmark_adapted.json` | 5,798 | `8e6985b2c2bd3ca1d12a59c56bdcbb6e177fccc84b5db147ce15cb1d989dfd9a` |
| `backend/model_artifacts/results/generalization_benchmark_baseline.json` | 5,898 | `d34218093cb5cf7fa2c5fe771ce1a4ea9228e321f6a730be66e22a2593fca4f3` |
| `backend/model_artifacts/results/generalization_comparison.json` | 17,163 | `13084ff88d51220e2d7db033b202bb26bcb49559fa6fda24bed95dcee290cabd` |
| `backend/model_artifacts/results/inference_benchmark.json` | 3,555 | `6f783e22f6d914e83cd83a8c15d83556c686426dbcb7dfaa754d919f0a293ff6` |
| `backend/model_artifacts/results/inference_benchmark.md` | 3,201 | `f40a1167ce0922f07f3b63324de03036c1aec9d64d8e14e4d8b75fbacae32095` |
| `backend/model_artifacts/results/test_evaluation_epoch46.json` | 2,929 | `8fd74972ef922ff3dd16dcda77ed56bbbbf987c670b0827d8b3a7863431fc266` |
| `backend/model_artifacts/results/test_evaluation_epoch46.md` | 3,651 | `e4791930749853738a4a535350e15068fcca25dfc12f4e0be38f8f99f24885ef` |
| `backend/model_artifacts/tokenizer/.gitkeep` | 53 | `8f93b9f0e105522eea4571018590d2cb1c2ae9e8b939b29a2d6c519b646af6f5` |
| `backend/model_artifacts/tokenizer/vocab.json` | 679,624 | `0a157b1e4debef738b297ef38403cf7df6bb5ead690d7558e6c3e19b9b3d0d17` |
| `backend/requirements.txt` | 181 | `a9b8b5114e7b2c3c94abf11bb80999a1b5d85887dd28ea23d152801ffad0f150` |
| `backend/tests/__init__.py` | 46 | `732996331d0d89c04cf9f02b9012415f92163a95e1f54681c006202f46c455cd` |
| `backend/tests/test_api.py` | 13,356 | `fe3301dec95bb1bdfc48e6ceb9d30cea18ca0326a04d95e1495fa6f967078211` |
| `backend/tests/test_evaluation.py` | 5,893 | `b245b5d4071483f65e35d09a127c8d7f6e8afabeaa149cf5f6cb6d0d02ac4be7` |
| `backend/tests/test_generalization.py` | 8,713 | `e0dcb61d52d1eb7ad3f01755b7f273ef3dd307b4cd636c6e34f5a89a07a5ed7c` |
| `backend/tests/test_pipeline.py` | 22,767 | `80359771baa752735032a798097eb75df95022f6fb359474e541ffc754324b10` |
| `backend/training/__init__.py` | 45 | `592ed5429be3f6a645c2939ddd329f71982bddd3a6944c6a496627c21f255124` |
| `backend/training/checkpoint.py` | 1,954 | `0ccef01299d161d5f7844815c4910a1a1e6106b4796b8b5fe735ffc0abd06c64` |
| `backend/training/dataset.py` | 3,410 | `63c809022d49ad8d7d1d9e963601ba660d1dc24eb565dfc291bde197b159bfd0` |
| `backend/training/download_corpus.py` | 4,314 | `d1261bd3f36cf987e921f920a35a650a00eee8da835c3108b777a559658bd301` |
| `backend/training/evaluate.py` | 3,978 | `25d0712ffac09e0c0bd2cb85c04429cd6a1a3b210367f1b500cecca17792e992` |
| `backend/training/evaluate_multidomain.py` | 17,341 | `6cd6d1bb4373daa50b37a2cfb8eae384d4bc1ef5658bf31085abb3a4a9bfee89` |
| `backend/training/generate_comparison_report.py` | 14,715 | `f6a1184c13357c0fc6d006a1cbddf087bbf4f3dc352f5190d57d2d56b2acae06` |
| `backend/training/pipeline.py` | 12,274 | `d56ee3d05ad967bada5ca445b9efc7fea256740a317333e893e3b99366d03428` |
| `backend/training/preprocess.py` | 7,418 | `845a07ba20a23b82264d6f5a766f8d281a4db096b43689846511b78f906807c2` |
| `backend/training/serialize.py` | 5,032 | `c19da1e5a3a4d7c3c95b4672cfee70c219fd56d1036701040b7e76ed11647eb5` |
| `backend/training/test_evaluate.py` | 27,917 | `623acdfda1614e73b5ad6cf6f24bfd1f1dc8b86e49f37cc4b5e1f12b4de6be24` |
| `backend/training/tokenize.py` | 2,426 | `0a9e9d79f8278165afb32067b89a39b85ab093b03ed47634f69ecb08c7ca09ea` |
| `backend/training/train.py` | 25,602 | `edd31cadf0b5b67b6f0ae120b3246b872d84ca4ea734a1469326b0519d91d43e` |
| `backend/training/train_generalization.py` | 9,327 | `39a8a54ba99c92601a12171548f90407e7d89bab74c7cae512692c8f3b8bcc30` |
| `backend/training/validate_pipeline.py` | 12,128 | `8880a641c94a9ba1c87f54522b80fbb8eed03c8bb417d62ee8f34eae2790d128` |
| `Data Pipeline.md` | 4,049 | `a999df3c49f60420a142fd508c8d6d566c423b613343a68a55cb69bc5d93e3eb` |
| `dataset.md` | 5,029 | `e19dd1826d9cf6e08f63d09c1a1500e4f3ac6be613ee027fab5e01f6db792a78` |
| `experiments/multidomain_v1/checkpoints/adaptation_training_summary.json` | 1,648 | `5183cdda1ebdaeabd121a5c424a2e75e8a038b64b0fec1ae38dcbd195dfe393f` |
| `experiments/multidomain_v1/checkpoints/multidomain_adapted.pt` | 106,996,080 | `a737ed30c48481dd78bf23e52a2ceb5eddeabb33508b6a0ea408afb82f21c875` |
| `experiments/multidomain_v1/configs/approach_a_word.json` | 1,695 | `36d037f5b2a346e9337ca26d6bccd01f1c49e8a74d943d999bb98b8d5209465c` |
| `experiments/multidomain_v1/configs/approach_b_bpe.json` | 1,661 | `902e5d2055a120e600cf3cf76e0af29e9a0516cecc6652d8e65d475e7890b682` |
| `experiments/multidomain_v1/configs/evaluation_allocation.json` | 12,406 | `bc13f101e9877cfe758f10d3e0eb4eb4f2fc73cace367b2e91eaf59b30f519e0` |
| `experiments/multidomain_v1/pilot/audit_pilot.py` | 20,058 | `796406df399aa2bb0c356c1fd5b71c3988768dab7f9634f7c8998f077ce9e56d` |
| `experiments/multidomain_v1/pilot/ingest_and_validate.py` | 60,523 | `fdb111379b176c5e2d0a63e62e8f9b485b603e199bd0654b2f5876e1811c4075` |
| `experiments/multidomain_v1/pilot/pilot_config.json` | 5,651 | `e1191277cb6381611e1d645a77c7dba0f0ca470265974d669f8632d1d7027a12` |
| `experiments/multidomain_v1/pilot/pilot_manifest.jsonl` | 145,188 | `49cd05f7f5aac661c943eb7f4b26724185a4b7c1704eb3f61b13de96fb547b06` |
| `experiments/multidomain_v1/pilot/processed/metadata.json` | 4,801 | `6abe4067b3f66acc61aabcc67844b4ea68387cdcaf5579c79c0ad072e5a121e6` |
| `experiments/multidomain_v1/pilot/processed/test.bin` | 42,990 | `873f9593293c1667dd94f9de07177bab0b10ac704a63aa685f64ddd225940288` |
| `experiments/multidomain_v1/pilot/processed/test_offsets.bin` | 144 | `ddfb71e8761bf3aa7ca184bd6be7bc95df2d11438dc934d6518a3085d9b2a850` |
| `experiments/multidomain_v1/pilot/processed/train.bin` | 164,268 | `06b1b38f53cbdeb9b5f05dba0547b1e62f9eef6abe7af596ff2dfc3772213109` |
| `experiments/multidomain_v1/pilot/processed/train_offsets.bin` | 728 | `f70b331763f8b4b58dd8fb4bb285d8e18ba929a81df5a5c7c163176f5dac5d0e` |
| `experiments/multidomain_v1/pilot/processed/val.bin` | 40,708 | `cdcd6ad7983bb6df0a96e5a7385058e39784e50a56bf3f587178432b64e704d9` |
| `experiments/multidomain_v1/pilot/processed/val_offsets.bin` | 160 | `719e91883ff7864fc59f0a62b6c2a33099b406f45e5f81645f4586ad9d13ead9` |
| `experiments/multidomain_v1/pilot/REAL_DATA_PILOT_REPORT.md` | 38,547 | `58a7b0f913c4b9c3fbdbe70c6a1984a0b856a7bcc3afd95cb272122f3b7fd702` |
| `experiments/multidomain_v1/validate_evaluation_plan.py` | 5,496 | `56d80737c68394c8c3aa56238f8a1be158203255f508d9767f04123ddbdb4da4` |
| `frontend/.gitignore` | 480 | `207e265ff4901f9ad9f96d8ce08530e04f9fc600815472a66d0a446096d654cd` |
| `frontend/AGENTS.md` | 678 | `63f2c50380ed6303237cce215ce27af1d620d094c215e28d1b1538a3c070e3bb` |
| `frontend/app/favicon.ico` | 25,931 | `2b8ad2d33455a8f736fc3a8ebf8f0bdea8848ad4c0db48a2833bd0f9cd775932` |
| `frontend/app/globals.css` | 577 | `26944526ba56cdcf9a091c799f21314339f3b6f6bcaffa9d82ee4696699a0320` |
| `frontend/app/layout.tsx` | 695 | `a51729a7f5c195acfc07d70322bf11cddb0b66f3e0b6b9c6103ea3d614bc87ba` |
| `frontend/app/page.tsx` | 6,765 | `98c439416b79ddbf3415fbefbe51fa033be5e7be97f7e14eb290931e557d0a0b` |
| `frontend/CLAUDE.md` | 11 | `336cc4fbf19beaada7ccf9986414fa91851a8d7a07dfb3ccbe800a69eed0ab49` |
| `frontend/components/GeneralizationSection.tsx` | 31,033 | `3ce43f5a1731491223f2d6f8411405e6f4c46971bcd2149246a997daf985b112` |
| `frontend/components/GenerationSection.tsx` | 10,248 | `e11e42e62853865f13da130b6aef8cac11052592be47a3faba07586e17d3865d` |
| `frontend/components/HistoryPanel.tsx` | 4,062 | `10e2e70bdf4539120854c3bebbd71ea42f42092c05332d3e54cfbf236c6d21e8` |
| `frontend/components/HowItWorks.tsx` | 5,003 | `9cde44000235bd37c2d5f98f2d0f464ea4a69139db6d3faabb2d82b8deda12cf` |
| `frontend/components/ModelInfoBadge.tsx` | 3,309 | `f7595bf7ce58bf3e6459ba9b67686af220cf61d0e0c066b8ecae733dad978fca` |
| `frontend/components/PredictionSection.tsx` | 7,858 | `4530ebbe52675b86bed53c4c924064f23fb039cd3f5211060b32c7ec90c73bbf` |
| `frontend/components/ThemeToggle.tsx` | 2,974 | `4feb46cd49eb906981db07ae3fa5543c0de27141197001df9462d762ef35fcc1` |
| `frontend/eslint.config.mjs` | 465 | `870f1adccecf3051cbcd9fd307cef51d7633cf510979c181a81f4b1797273493` |
| `frontend/lib/api.ts` | 7,245 | `be60dc0482d43409ad0a894243604b395ca4dd6f680bba6588058ab8a4aaec62` |
| `frontend/next-env.d.ts` | 296 | `0f70629890b72a0a82e91972cc032c04b658b26c265373cb711cf576bfbf8fcc` |
| `frontend/next.config.ts` | 129 | `f1971e276440f17f7fbd1e3f0c0a297b3632783f30a14a10f1010779b5afd2fe` |
| `frontend/package-lock.json` | 236,690 | `4a0938421c13383fce13796d7d75011f4b784e2fa9d5310bea82423618a69028` |
| `frontend/package.json` | 579 | `237812aba3c1b466247e16dbf09a3a077156b43639f126aaff16298049fb0485` |
| `frontend/postcss.config.mjs` | 94 | `dfac7ac2d86d326a0e5adb024e7943c181393ed17a5fcb8f0315b24c7da6ddde` |
| `frontend/public/file.svg` | 391 | `2b67812c325c199a02536cdbeea0c593a72f707d323b72ee3e08dbab06753bd4` |
| `frontend/public/globe.svg` | 1,035 | `b614b9bf183925957661ac851498fe1d8029fd43a62fbfed86f9e2624a57e7cf` |
| `frontend/public/next.svg` | 1,375 | `55995dfad6ecb4945a1e856ddca03c5e16aa5bf13fd21b4df6a74ae79357bcfc` |
| `frontend/public/vercel.svg` | 128 | `f081337b2fee635b455b63275406a3e7f39d6a014e25ad90dab5a67e62a12ac4` |
| `frontend/public/window.svg` | 385 | `644768c4aaeb4767bce293344eeb0c125fb804a94d801440424072202d85e3a1` |
| `frontend/README.md` | 1,450 | `60b55ff7df79af72590f9524208e46642bc32bdc175cdad41349681c0e2f958f` |
| `frontend/tests/smoke.test.mjs` | 10,172 | `c094ceeb99d72bc1d9811753e7df77939c8c8b8d6622db850dcdae63050dd139` |
| `frontend/tsconfig.json` | 666 | `be18523b23b78b6e1a876ddd107d330f0168bddf09d751b57e0b5edce69c66f3` |
| `PRD.md` | 3,334 | `befb5ac328e14d0a152e0c9ece556e182ff9bcaf5c0b7c0f4f25b76c90b33415` |
| `README.md` | 20,347 | `b59437b33ce56d6bf8663b83532ad2234835f984379f67050db338a02a1e98be` |
| `scripts/package_release.py` | 10,184 | `d1fbdbd8edf559b05f80a720d5c5174a1b3d893d31e97b6f3e9d37efa5d90a22` |
| `scripts/verify_checkpoints.py` | 8,796 | `0432790f3c80e7f85735abecdb20a2f9476dc2db346cb05aa0879013ebd253d0` |
