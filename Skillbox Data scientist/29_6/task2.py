
# Функция принимает на вход строку, которая
# состоит из скобок трех типов: круглые, квадратные
# и фигурные. Функция должна проверить, является ли
# переданная последовательность скобок сбалансированной
# или нет. Функция возвращает True, если последовательность
# сбалансирована, и False – в противном случае.
def is_balanced(string):
    pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
            }
    stack = []
    for char in string:
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    return not stack


def test_is_balanced():
    cases = {
        '((((((((())))))))': False,
        '{[()]}{{}}': True,
        '{[()]}{]{}}': False,
        '{{{((([[[]]])))}}}': True,
        '{}': True,
        '(': False,
        '(}': False,
        '(((())))[]{}': True,
        '((()': False,
        '[{}{})(]': False,
        '{[]{([[[[[[]]]]]])}}': True,
        '{[]{([[[[[[]])]]])}}': False,
    }
    for i, case in enumerate(cases.keys()):
        if is_balanced(case) == cases[case]:
            print(f'{i}: OK')
        else:
            print(f'{i}: Not OK')


def main():
    test_is_balanced()


if __name__ == '__main__':
    main()