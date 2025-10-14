class ImprovementChecker:
    """
    class used to determine if there's an improvement in network performance by checking loss and accuracy values
    """
    def __init__(self, db, lfi):
        self.db = db
        self.lfi = lfi

    def checker(self, val_acc, val_loss):
        """
        method that analyses trends in metrics and determines whether there was an improvement
        :param val_acc val_loss: current accuracy and loss values used to compare with values from past iterations
        :return: boolean indicating if there was an improvement
        """
        # obtain the values from db
        score, acc = self.db.get()

        # if the length of the score history is 0, nothing has been saved yet in the db
        if len(score) == 0:
            return None

        # set to true that there was an improvement in socre and accuracy
        score_check = True
        acc_check = True

        # if there's a degradation compared to the last training
        if val_acc < score[len(score) - 1]:
            score_check = False
        
        return score_check