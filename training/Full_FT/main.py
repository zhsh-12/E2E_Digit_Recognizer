from training.Full_FT.model import Digit_recognizer
from training.Full_FT.dataloader import get_dataloaders
from training.Full_FT.trainer import set_device, train_model, test_model
from training.Full_FT.experiments import ExperimentLogger
from training.Full_FT.args import args_parse
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
        args.mode = 'FineTuning' if args.mode == 'FineTuning' else 'AutoML'
        
        #obtain dataloader
        new_train_loader, new_val_loader, new_test_loader, old_train_loader, _, old_test_loader = get_dataloaders(args)

        #set device
        device = set_device()
        print(f"Using device: {device}")

        #file_save path setting
        exp_dir = base_dir / "model_output" / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        resume_path = exp_dir / 'last.pth' if (args.mode == 'FineTuning' and (exp_dir / 'last.pth').exists()) else None
        config_path = exp_dir / "config.json"
        metrics_path = exp_dir / "metrics.json"

        log_path = log_dir / "tensorboard" / exp_id
        log_path.mkdir(parents=True, exist_ok=True)

        model_path = base_dir / "model_output"

        #record experiment settings
        config_data = vars(args).copy()
        config_data.update({
            "exp_id": exp_id,
            "device": str(device)
        })
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        #define model framework、load model parameters
        model = Digit_recognizer()
        model.to(device) 

        checkpoint = torch.load(model_path /'old_model.pth', map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)


        #start training
        train_loss, val_loss, train_acc, val_acc = train_model(model, new_train_loader, old_train_loader, new_val_loader,
                                                            device=device, epochs=args.epochs, lr=args.lr, 
                                                            resume_path=resume_path,log_path=log_path, output_dir=exp_dir)

        #obtain best model
        checkpoint = torch.load(exp_dir /'best.pth', map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        #evaluate model
        new_test_loss, new_test_acc, old_test_loss, old_test_acc = test_model(model, new_test_loader, old_test_loader, device=device, png_path=exp_dir)

        #experiment results
        results = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'new_test_loss': new_test_loss,
                'old_test_loss': old_test_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'new_test_acc': new_test_acc, 
                'old_test_acc': old_test_acc, 
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

