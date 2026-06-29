from training.initial_training.model import Digit_recognizer
from training.initial_training.dataloader import get_dataloaders
from training.initial_training.trainer import set_device, train_model, test_model
from experiments import ExperimentLogger
from args import args_parse
import torch, json
from pathlib import Path

base_dir = Path(__file__).resolve().parent
log_dir = base_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

def main():
    #initialize experiments
    logger = ExperimentLogger(log_file=f'{log_dir}/Experiments.csv')
    exp_id = logger.generate_exp_id()
    logger.start()
    args = None 
    exp_dir = None 

    try:
        #resolve experiment input settings
        args = args_parse()
        args.mode = 'baseline' if args.mode == 'baseline' else 'autoML'
        
        #obtain dataloader
        train_loader, val_loader, test_loader = get_dataloaders(args)

        #set device
        device = set_device()
        print(f"Using device: {device}")

        #define model
        model = Digit_recognizer()
        model.to(device) 

        #file_save path setting
        exp_dir = base_dir / "model_output" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        resume_path = exp_dir / 'last.pth' if (args.mode == 'baseline' and (exp_dir / 'last.pth').exists()) else None
        config_path = exp_dir / "config.json"
        metrics_path = exp_dir / "metrics.json"

        log_path = log_dir / "tensorboard" / exp_id
        log_path.mkdir(parents=True, exist_ok=True)

        #record experiment settings
        config_data = vars(args).copy()
        config_data.update({
            "exp_id": exp_id,
            "device": str(device)
        })
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        #start training
        train_loss, val_loss, train_acc, val_acc = train_model(model, train_loader, val_loader, epochs=args.epochs, 
                                                            lr=args.lr, device=device, resume_path=resume_path,
                                                            log_path=log_path, output_dir=exp_dir)

        #obtain best model
        checkpoint = torch.load(exp_dir /'best.pth', map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        #evaluate model
        test_loss, test_acc, test_source_acc = test_model(model, test_loader, device=device, png_path=exp_dir)

        #experiment results: source_map = {0: "MNIST", 1: "EMNIST", 2: "SVHN"}
        results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                "mnist_acc": test_source_acc.get(0),
                "emnist_acc": test_source_acc.get(1),
                "svhn_acc": test_source_acc.get(2),  
                'exp_dir': str(exp_dir),
                'error': None
            }
       
        logger.log(args, results, exp_id=exp_id)

        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=4)

    except Exception as e:
        logger.log(
            args if args else {}, 
            {"error": str(e), "exp_dir": str(exp_dir) if exp_dir else None}, 
            exp_id=exp_id
        )
        raise

if __name__ == '__main__': 
    main()

