# MFLP

## Introduction

This is code of MFLP(“Multi-space feature learning enhanced microbe-drug association prediction”).

Operating environment
```markdown
torchvision == 0.10.0+cu111
torch == 1.9.0+cu111
python == 3.8.10
numpy == 1.23.5
scikit-learn == 1.1.3
scipy == 1.10.0
pandas == 1.5.2
matplotlib == 3.4.3
```

Data description
```markdown
adj: known adjacent matrix for drugs and microbes, i.2., interaction.
microbe_features: contains feature vectors of 173 microbes.
drug_names: contains names of 1373 drugs.
microbe_names: contains names of 173 microbes.
drugsimilarity: similarity between drugs.
hyperbolicity.py:calculate the σ-hyperbolicity value of microbe-drug network 
```
Run steps
```markdown
1.install operating environment.
2.run main.py.
```

