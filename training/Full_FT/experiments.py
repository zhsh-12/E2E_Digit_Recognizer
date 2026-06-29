import csv, os, time
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.fieldnames = [
            'exp_id', 'date', 'sample_ratio', 'batch_size', 'lr',  'epochs', 
            'total_time', 'mode', 'error','exp_dir', 
            'train_loss', 'val_loss', 'new_test_loss', 'old_test_loss',
            'train_acc', 'val_acc', 'new_test_acc', 'old_test_acc'
        ]
        #create file if not exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        self.start_time = None
    
    def start(self):
        self.start_time = datetime.now() 
    
    def __get_total_time(self):
        if self.start_time is None:
            return None
        return (datetime.now() - self.start_time).total_seconds()

    def generate_exp_id(self):
        if not os.path.exists(self.log_file):
            return 'Exp_001'
        with open(self.log_file, mode='r') as f:
            rows = list(csv.DictReader(f))
            if not rows:
                return 'Exp_001'
            last_id = rows[-1]['exp_id']
            try:
                num = int(last_id.split('_')[1]) + 1
            except:
                num = len(rows) + 1
            return f"Exp_{num:03d}"
        
    def log(self, args, results, exp_id=None):
        if isinstance(args, dict):
            config = args.copy()
        elif args is None:
            config = {}
        else:
            config = vars(args).copy()
        
        if exp_id is None:
            exp_id = self.generate_exp_id()

        row = {
            'exp_id': exp_id,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            #experiment input settings
            'sample_ratio': config.get('sample_ratio'),
            'batch_size': config.get('batch_size'),
            'lr': config.get('lr'),
            'epochs': config.get('epochs'),
            'total_time': self.__get_total_time(),
            'mode': config.get('mode'),
            #experiment output results
            'error': results.get('error'),
            'exp_dir': results.get('exp_dir'),
            'train_loss': results.get('train_loss'),
            'val_loss': results.get('val_loss'),
            'new_test_loss': results.get('new_test_loss'),
            'old_test_loss': results.get('old_test_loss'),
            'train_acc': results.get('train_acc'),
            'val_acc': results.get('val_acc'),
            'new_test_acc': results.get('new_test_acc'),
            'old_test_acc': results.get('old_test_acc')
        }
        for _ in range(3):
            try:
                with open(self.log_file, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerow(row)
                print(f'\n ✅ Experiment logged: {exp_id}')
                break
            except:
                time.sleep(0.1)

