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
        acc, loss = self.db.get()

        # if the length of the accuracy history is 0, nothing has been saved yet in the db
        if len(acc) == 0:
            return None

        # set to true that there was an improvement in loss and accuracy
        acc_check = True
        loss_check = True

        # if there's a degradation compared to the last training
        if val_acc < acc[len(acc) - 1]:
            acc_check = False
        '''

        # iterate over all accuracy values and, in case there was a degradation,
        # set the boolean 'acc_check' to false
        for a in acc:
            if val_acc < a:
                acc_check = False
                break
        
        # the same applies to loss values
        for l in loss:
            if val_loss > l:
                loss_check = False
                break
        '''
        return acc_check