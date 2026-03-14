```mermaid
flowchart TB
    subgraph Data["DATA LAYER"]
        direction TB
        CSV[("CSV Data<br/>IPFIX ML Instances")]
        JSON[("JSON Data<br/>IPFIX Records")]
        
        subgraph Preprocessors["Preprocessors"]
            CSVPreproc["CSVPreprocessor<br/>(preprocessor.py)"]
            JSONPreproc["JSONPreprocessor<br/>(json_preprocessor.py)"]
        end
        
        Features[("Features<br/>Extraction")]
        Scaler[("StandardScaler")]
        LabelEnc[("LabelEncoder")]
    end

    subgraph Sequences["Sequence Creation"]
        SeqCreator["SequenceCreator<br/>create_sequences_with_stride"]
        Train["Train Set"]
        Val["Validation Set"]
        Test["Test Set"]
    end

    subgraph Models["MODEL LAYER"]
        direction LR
        LSTM["LSTMClassifier"]
        Transformer["TransformerClassifier"]
        CNNLSTM["CNN-LSTM<br/>Classifier"]
        CNN["CNN Classifier"]
    end

    subgraph Adversarial["ADVERSARIAL LAYER - Existing"]
        direction TB
        FeatureAtt["FeatureLevelAttack<br/>(IoT-SDN Style)"]
        SeqAtt["SequenceLevelAttack<br/>(PGD/FGSM + BPTT)"]
        HybridAtt["HybridAdversarialAttack<br/>60% Clean + 20% Feature + 20% Sequence"]
        Evaluator["AdversarialEvaluator"]
    end

    subgraph Training["TRAINING LAYER"]
        Trainer["AdversarialTrainer"]
        Phase1["Phase 1: Clean<br/>60% epochs"]
        Phase2["Phase 2: Feature-Level<br/>20% epochs"]
        Phase3["Phase 3: Sequence-Level<br/>20% epochs"]
    end

    subgraph Security["SECURITY LAYER - To Implement"]
        direction TB
        
        subgraph Validator["VALIDATOR (New)"]
            ValInput["InputValidator"]
            ValFeature["FeatureValidator"]
            ValSeq["SequenceValidator"]
            ValOutput["Valid Output<br/>✓ Reject / ✗ Accept"]
        end
        
        subgraph Discriminator["DISCRIMINATOR (New)"]
            DirDis["DirectionDiscriminator<br/>Detects pkt_dir manipulation"]
            StatDis["StatisticalDiscriminator<br/>Checks feature anomalies"]
            TempDis["TemporalDiscriminator<br/>Verifies sequence consistency"]
            DisOutput["Discrimination Result<br/>Real vs Adversarial"]
        end
        
        subgraph AdaptiveEngine["ADAPTIVE ENGINE (New)"]
            AE_Monitor["AttackMonitor<br/>Real-time detection"]
            AE_Strategy["DefenseStrategy<br/>Adaptative response"]
            AE_Update["ModelUpdater<br/>Online learning"]
            AE_Feedback["FeedbackLoop"]
        end
    end

    subgraph Output["OUTPUT LAYER"]
        Results["Results.json"]
        History["History.json"]
        ModelsSaved["Saved Models<br/>best_model.pt"]
    end

    %% Data Flow
    CSV --> CSVPreproc
    JSON --> JSONPreproc
    CSVPreproc --> Features
    JSONPreproc --> Features
    Features --> Scaler
    Features --> LabelEnc
    Scaler --> SeqCreator
    SeqCreator --> Train
    SeqCreator --> Val
    SeqCreator --> Test

    %% Model Flow
    Train --> Models
    Models --> LSTM
    Models --> Transformer
    Models --> CNNLSTM
    Models --> CNN

    %% Adversarial Flow
    Train --> Adversarial
    Models --> FeatureAtt
    Models --> SeqAtt
    FeatureAtt --> HybridAtt
    SeqAtt --> HybridAtt
    HybridAtt --> Trainer
    Trainer --> Phase1
    Trainer --> Phase2
    Trainer --> Phase3

    %% Security Integration (New Components)
    Models -.-> Validator
    FeatureAtt -.-> Validator
    SeqAtt -.-> Validator
    Validator --> Discriminator
    Discriminator --> AdaptiveEngine
    AdaptiveEngine -->|Update| Models
    AdaptiveEngine -->|Feedback| Trainer
    
    %% Output
    Phase1 --> Output
    Phase2 --> Output
    Phase3 --> Output
    AdaptiveEngine --> Output

    %% Styling
    classDef existing fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef new fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef data fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef model fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    
    class CSV,JSON,Features,Scaler,LabelEnc data
    class LSTM,Transformer,CNNLSTM,CNN model
    class FeatureAtt,SeqAtt,HybridAtt,Evaluator existing
    class Trainer,Phase1,Phase2,Phase3 existing
    class Validator,Discriminator,AdaptiveEngine new
```

## Architecture Details

### Composants Existants:
- **Data Layer**: CSV/JSON preprocessing, feature extraction, scaling
- **Sequence Creation**: Windowing with stride for temporal sequences
- **Models**: LSTM, Transformer, CNN-LSTM, CNN classifiers
- **Adversarial Layer**: Feature-level, Sequence-level, Hybrid attacks
- **Training**: 3-phase adversarial training (60% clean + 20% feature + 20% sequence)

### Composants à Implémenter:

| Composant | Rôle |
|-----------|------|
| **Validator** | Valide les entrées avant inference - détecte les features invalides |
| **Discriminator** | Distingue les échantillons réels des adversariaux |
| **Adaptive Engine** | Adaptation en temps réel aux attaques détectées |

Voulez-vous que je génère le code de ces composants?
