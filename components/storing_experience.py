import sqlite3 as db
from sqlite3 import Error
import re


class StoringExperience:
    """
    Class used to interface with the db and in which to store accuracy/loss values and tuning informations
    """
    def __init__(self):
        """
        initialise attributes in which to store into the db operations of creation and cleaning of tables
        """
        self.db_name = "database/experience.db"
        self.destroy1 = """
            DROP TABLE IF EXISTS ranking
        """
        self.destroy2 = """
            DROP TABLE IF EXISTS experience
        """
        self.create1 = """
            CREATE TABLE IF NOT EXISTS ranking (
            id integer PRIMARY KEY,
            val_acc real,
            val_loss real)
        """
        self.create2 = """
            CREATE TABLE IF NOT EXISTS experience (
            id integer PRIMARY KEY,
            evidence text)
        """

    def connection(self):
        """
        method used to interface with the database
        :return: connection with db
        """
        conn = None
        try:
            conn = db.connect(self.db_name)
            return conn
        except Error as e:
            print(e)
        return conn

    def create_db(self):
        """
        method used for creating the necessary tables in which to store the informations,
        executing the initially defined transactions
        """
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute(self.destroy1)
            c.execute(self.destroy2)
            c.execute(self.create1)
            c.execute(self.create2)
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_ranking(self, val_acc, val_loss):
        """
        method used to insert accuracy and loss values into the db
        """
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO ranking (val_acc, val_loss) VALUES (' + str(val_acc) + ',' + str(val_loss) + ')')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def insert_evidence(self, evidence):
        """
        method used to insert evidences into the db.
        each evidence has anomaly of the network, a possible solution and a
        boolean that indicates if there was an improvement
        """
        conn = self.connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO experience (evidence) VALUES ("' + str(evidence) + '")')
        except Error as e:
            print(e)
        conn.commit()
        conn.close()

    def formatting(self, res):
        """
        method used to split loss and accuracy values into two lists
        :param res: list of lists, each of which will contain two values, one of acc and one of loss
        :return: two lists containing loss and acc values
        """
        acc = []
        loss = []
        for i in res:
            acc.append(i[1])
            loss.append(i[2])
        return acc, loss

    def get(self):
        """
        method used to obtain acc and loss values stored in the db
        :return: list of loss and acc values
        """
        conn = self.connection()
        c = conn.cursor()
        c.execute("SELECT * FROM ranking")
        res = self.formatting(c.fetchall())
        conn.close()
        return res


if __name__ == '__main__':
    se = StoringExperience()
    se.create_db()
    se.insert_ranking(0.7225, 0.7423)
    se.insert_ranking(0.7325, 0.7113)
    se.insert_ranking(0.7895, 0.4353)
    acc, loss = se.get()

    print(acc)
    print()
    print(loss)
