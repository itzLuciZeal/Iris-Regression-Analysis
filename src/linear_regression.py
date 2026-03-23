import numpy as np
import math

class Inference:
    def __init__(self, x: list, y: list) -> None:
        '''Important inputs of x and y; the length of y and x should be equal'''
        if len(x) != len(y): raise ValueError("Both lists must be the same length")
        self.n = len(x)
        self.x = np.array(x)
        self.y = np.array(y)

        # Sum of each given data
        self.sum_x = self.x.sum()
        self.sum_y = self.y.sum()
        self.sum_x_2 = (self.x**2).sum()
        self.sum_y_2 = (self.y**2).sum()
        self.sum_xy = (self.x * self.y).sum()


    def pearson_r(self) -> float:
        '''Standard way to get pearson's r'''
        numerator = self.n*self.sum_xy - (self.sum_x*self.sum_y)
        denominator = np.sqrt((self.n*self.sum_x_2 - (self.sum_x)**2) * (self.n*self.sum_y_2 - (self.sum_y)**2))
        if denominator == 0: return 0.0

        # Clip the result to ensure it stays between -1 and 1
        r = numerator / denominator
        if denominator == 0: return 0.0

        return np.round(np.clip(r, -1.0, 1.0), 4)
   

    def methods_of_diff(self) -> float:
        '''Includes the differences of x and y to calculate pearson's r'''
        d = self.x - self.y
        d_2 = d**2
        _sum_d = d.sum()
        _sum_d_2 = d_2.sum()

        # Variance
        Sx_2 = (self.sum_x_2 - ((self.sum_x)**2/self.n))/self.n
        Sy_2 = (self.sum_y_2 - ((self.sum_y)**2/self.n))/self.n
        Sd_2 = (_sum_d_2 - ((_sum_d)**2/self.n))/self.n

        # Standard Deviation
        print(Sy_2, Sd_2)
        Sx, Sy, Sd = math.sqrt(max(0, Sx_2)), math.sqrt(max(0, Sy_2)), math.sqrt(max(0, Sd_2))
        if Sx * Sy == 0: return 0.0

        # Solve
        numerator = Sx_2 + Sy_2 - Sd_2
        denominator = 2*(Sx)*(Sy)
        if denominator == 0: return 0.0

        return np.round(numerator/denominator, 4)
    

    def _linear_reg_coefficient(self):
        '''Getting the slope (m) and y-intercept (b) of the given x and y data points'''
        num_M = (self.n)*(self.sum_xy) - (self.sum_x)*(self.sum_y)
        denom_M = (self.n)*(self.sum_x_2) - (self.sum_x)**2
        m = num_M/denom_M
        b = (self.sum_y - m*(self.sum_x))/self.n

        return m, b
    
    def linear_equation_text(self):
        m, b = self._linear_reg_coefficient()
        sign = "-" if b < 0 else "+"

        return f"y = {m:.2f}x {sign} {abs(b):.2f}"


    def reg_line_array(self):
        '''Returning the array of x array of values and y array of values of the given equation of regression line'''
        m, b = self._linear_reg_coefficient()
        x_lin = np.linspace(np.min(self.x), np.max(self.x), 100)
        y_lin = m * x_lin + b

        return x_lin, y_lin