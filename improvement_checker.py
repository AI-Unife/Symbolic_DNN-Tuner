class ImprovementChecker:
    def __init__(self, db, lfi):
        self.db = db
        self.lfi = lfi

    def checker(self, val_acc, val_loss):
        acc, loss = self.db.get()
        acc_check = True
        loss_check = True
        for a in acc:
            if val_acc < a:
                acc_check = False
                break
        for l in loss:
            if val_loss > l:
                loss_check = False
                break
        if not acc_check:
            return False
        else:
            return True