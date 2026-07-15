import math


class jump:
    @staticmethod
    def activate(x):
        if x >= 1:
            return 1
        else:
            return 0

    @staticmethod
    def derivative(x):
        return 0


class Relu:
    @staticmethod
    def activate(x):
        if x < 0:
            return 0
        elif x >= 0 and x <= 1:
            return x
        else:
            return 1


    @staticmethod
    def derivative(x):
        return 1


class equal:
    @staticmethod
    def activate(x):
        return x

    @staticmethod
    def derivative(x):
        return 1