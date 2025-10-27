# Enhanced Multi-Disease Detection System

## Overview

This enhanced version of the H. pylori CDSS includes **proper clinical decision support** capabilities with:

- **Multi-Disease Detection**: Detects 5 different GI conditions
- **Enhanced Action Space**: Navigation + Diagnostic actions
- **Sophisticated Reward System**: Rewards for correct detection, penalties for errors
- **Clinical Metrics**: Sensitivity, Precision, F1 Score

---

## 🏥 Detected Conditions

The enhanced CNN can identify:

1. **Normal Tissue** - Healthy gastric mucosa
2. **H. pylori Gastritis** - Bacterial infection indicators
3. **Peptic Ulcer** - Erosive lesions
4. **Gastric Tumor** - Early-stage neoplasms
5. **Inflammation** - Non-specific inflammatory changes

---

## 🎮 Enhanced Action Space

### Navigation Actions (0-7)
- `0`: Rotate camera left (yaw+)
- `1`: Rotate camera right (yaw-)
- `2`: Tilt camera up (pitch+)
- `3`: Tilt camera down (pitch-)
- `4`: Move forward
- `5`: Move backward
- `6`: Zoom in
- `7`: Zoom out

### Diagnostic Actions (8-11)
- `8`: **FLAG_REGION** - Flag current view as abnormal
- `9`: **TAKE_BIOPSY** - Take high-resolution snapshot
- `10`: **REQUEST_AI** - Request CNN inference on current frame
- `11`: **DONE** - Complete examination

---

## 🎯 Reward System

### Positive Rewards
- **True Positive**: +2.0 × severity (correctly flag disease)
- **Coverage**: +0.05 per new area visited
- **Efficiency**: +0.01 per step saved
- **Completion Bonus**: +5.0 (if good performance)
- **Biopsy Near Disease**: +0.5 × severity

### Penalties
- **False Positive**: -0.5 (flag normal tissue as abnormal)
- **False Negative**: -1.0 × severity (miss disease)
- **Collision**: -1.0 (hit tissue wall)
- **Redundant Flagging**: -0.1 (flag same region twice)
- **AI Request**: -0.05 (encourage efficiency)

### Total Reward Formula
```
R = α × R_coverage + β × R_detection + γ × R_collision + δ × R_efficiency

Where:
  α = 0.3  (coverage importance)
  β = 0.6  (detection is PRIMARY)
  γ = -0.3 (collision penalty)
  δ = 0.1  (efficiency bonus)
```

---

## 📊 Clinical Metrics

The system tracks:

- **Sensitivity (Recall)**: TP / (TP + FN) - % of diseases correctly detected
- **Precision**: TP / (TP + FP) - % of flagged regions that are truly abnormal
- **F1 Score**: Harmonic mean of precision and sensitivity
- **Coverage**: % of GI tract examined
- **Efficiency**: Steps used vs max steps

### Performance Targets
- Sensitivity: > 95% (minimal missed lesions)
- Precision: > 80% (acceptable false positive rate)
- F1 Score: > 0.85
- Coverage: > 70%

---

## 🚀 Usage

### 1. Using the Enhanced Environment

```python
from backend.sim.env_enhanced import EndoscopyEnvEnhanced

# Create environment
env = EndoscopyEnvEnhanced(
    gltf_path="path/to/model.gltf",
    curriculum_stage="medium",
    use_cnn=True,
)

# Reset
obs, info = env.reset()

# Run episode
done = False
while not done:
    # Agent chooses action
    action = agent.predict(obs)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Print diagnostic metrics
    print(f"TP: {info['true_positives']}, FP: {info['false_positives']}")
    print(f"Sensitivity: {info['sensitivity']:.2f}, Precision: {info['precision']:.2f}")
```

### 2. Training RL Agent with Enhanced Environment

```bash
# Train PPO agent on enhanced environment
python -m backend.models.rl.train_rl_enhanced \
    --env-id EndoscopyEnhanced-v0 \
    --total-timesteps 1000000 \
    --reward-threshold 20.0 \
    --use-cnn
```

### 3. Training Enhanced Multi-Class CNN

```bash
# Train multi-disease classifier
python -m backend.models.cnn.train_cnn_enhanced \
    --data-dir data/synthetic_frames \
    --output-dir outputs/cnn_multiclass \
    --epochs 50 \
    --batch-size 32 \
    --use-focal-loss
```

### 4. Inference with Enhanced CNN

```python
from backend.models.cnn.model_enhanced import create_model
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = create_model(pretrained=True)
model.load_state_dict(torch.load('outputs/cnn_multiclass/best_model.pt'))

# Load image
image = Image.open('frame.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image_tensor = transform(image)

# Predict
disease, confidence, probs = model.predict_single(image_tensor)

print(f"Prediction: {disease} ({confidence:.2%} confidence)")
print(f"All probabilities: {probs}")
```

---

## 🧪 Evaluation

### Automated Testing

```bash
# Run evaluation on test set
python -m backend.models.rl.eval_enhanced \
    --model-path outputs/rl_policy/best_model.zip \
    --n-episodes 100 \
    --render
```

### Metrics Output

```
=== Clinical Performance Metrics ===

Detection Performance:
  Sensitivity:    96.5%
  Precision:      84.2%
  F1 Score:       89.9%
  
Disease-Specific:
  H. pylori:      95.0% sensitivity
  Ulcer:          98.2% sensitivity
  Tumor:          94.1% sensitivity
  Inflammation:   97.3% sensitivity
  
Navigation Performance:
  Coverage:       78.5%
  Collision Rate: 2.3%
  Avg Steps:      342 / 500
  
Overall Score:    85.3 / 100
```

---

## 🔬 Research Applications

### 1. Curriculum Learning
Train agents progressively:
```python
stages = ['easy', 'medium', 'hard']
for stage in stages:
    env = EndoscopyEnvEnhanced(curriculum_stage=stage)
    train_agent(env, stage)
```

### 2. Domain Randomization
Improve generalization:
- Random lighting conditions
- Variable disease presentations
- Different tissue textures
- Camera noise injection

### 3. Sim-to-Real Transfer
Bridge simulation to real endoscopy:
- Photorealistic rendering
- Physics-based lighting
- Anatomically accurate models
- Validated disease appearances

---

## 📈 Comparison: Basic vs Enhanced

| Feature | Basic System | Enhanced System |
|---------|-------------|-----------------|
| **Actions** | 9 (navigation only) | 12 (navigation + diagnostic) |
| **Disease Types** | Binary (lesion/no lesion) | 5 classes (multi-disease) |
| **Reward Focus** | Coverage (50%) | Detection (60%) |
| **Clinical Metrics** | None | Sensitivity, Precision, F1 |
| **Diagnostic Actions** | None | Flag, Biopsy, AI Request |
| **False Positive Penalty** | No | Yes (-0.5) |
| **Efficiency Tracking** | No | Yes (+0.01/step) |
| **Multi-Class CNN** | No | Yes (ResNet18 + Attention) |

---

## 🎓 Training Strategy

### Phase 1: CNN Pre-training
1. Generate synthetic dataset (100K frames)
2. Train multi-class CNN with Focal Loss
3. Achieve >85% validation accuracy
4. Export TorchScript model

### Phase 2: RL Policy Training
1. Load pre-trained CNN
2. Train PPO agent with CNN-in-the-loop
3. Use curriculum learning (easy → hard)
4. Target F1 score > 0.85

### Phase 3: Fine-tuning
1. Joint CNN + RL fine-tuning
2. Domain randomization
3. Adversarial examples
4. Human expert validation

---

## ⚠️ Important Notes

### Research Use Only
- **NOT a medical device**
- **NOT for clinical diagnosis**
- **NOT FDA approved**
- For research and educational purposes only

### Limitations
- Simulated environment (not real endoscopy)
- Synthetic disease presentations
- No patient data used
- Requires validation on real data

### Ethical Considerations
- Always validate with medical experts
- Never deploy without clinical trials
- Maintain human oversight
- Respect patient privacy

---

## 📚 References

1. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
3. **ResNet**: He et al., "Deep Residual Learning for Image Recognition" (2016)
4. **Curriculum Learning**: Bengio et al., "Curriculum Learning" (2009)

---

## 🤝 Contributing

To improve the enhanced system:

1. **Add Disease Types**: Extend `DISEASE_TYPES` in `env_enhanced.py`
2. **Improve Rewards**: Tune reward coefficients in `configs/sim.yaml`
3. **Better CNN**: Try EfficientNet, Vision Transformer
4. **Real Data**: Integrate with actual endoscopy datasets (with proper ethics approval)

---

## 📞 Support

For questions about the enhanced system:
- Open an issue on GitHub
- Check `PROJECT_SUMMARY.md` for architecture details
- Review `QUICKSTART.md` for setup help

---

**Last Updated**: 2025-10-27
**Version**: 2.0 (Enhanced Multi-Disease Detection)

