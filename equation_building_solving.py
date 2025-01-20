# from symbol_recognition import eqn,final_eqn
import sympy as sym
from sympy import *


def build_and_solve(eqn):
    def solve_equation(eqn):
        try:
            variable_list = ['a','b','c','y']
            eqn_var_list = []
            if 'y' in eqn or 'a' in eqn or 'b' in eqn or 'c' in eqn:
                        for var in variable_list:
                            if var in eqn: 
                                eqn_var_list.append(var)
                        # print(eqn_var_list)
                        if len(eqn_var_list) == 2:
                            vars_sym = symbols(' '.join(eqn_var_list))
                            if '=' in eqn: 
                                texts = eqn.split("=")
                                # print(texts)
                                equation = Eq(sympify(texts[0]), sympify(texts[1]))
                            else:
                                equation = Eq(sympify(eqn),0)
                            solution_1 = solve(equation, vars_sym[0])
                            solution_2 = solve(equation, vars_sym[1])


                            result = f"{eqn_var_list[0]} = {solution_1[0]}, {eqn_var_list[1]} = {solution_2[0]}"

                            
                        elif len(eqn_var_list) == 1:
                            vars_sym = symbols(eqn_var_list[0])
                            if '=' in eqn: 
                                texts = eqn.split("=")
                                # print(texts)
                                equation = Eq(sympify(texts[0]), sympify(texts[1]))
                            else:
                                equation = Eq(sympify(eqn),0)
                            solution_1 = solve(equation, vars_sym)
                            result = f"{eqn_var_list[0]} = {solution_1[0]}"

                        else:
                            result = "Too many variables"
                        return result

            else:
                    result = sympify(eqn)
                    return result

        except (ValueError, TypeError, AttributeError, RuntimeError, SyntaxError) as e:
            # print(f"Error: {e}")
            # print(e)
            return str('Wrong equation prediction or Invalid Equation')
        
    result = solve_equation(eqn)

    # print("Predicted Equation :" ,final_eqn)
    # print("Evaluated Answer :",solve_equation(eqn))
    return result



