import random

# Создание и заполнение таблицы Product
with open('create_and_fill_Product.sql', 'w') as file:
    file.write(
        'CREATE TABLE Product (maker CHAR(1), model INTEGER, type VARCHAR(10));\n'
    )
    file.write('INSERT INTO Product\n')
    file.write('VALUES\n')

    product_values = []
    for i in range(1, 101):
        maker = random.choice(['A', 'B', 'C', 'D', 'E'])
        model = random.randint(1000, 9999)
        product_type = random.choice(['PC', 'Laptop', 'Printer'])
        product_values.append(f'(\'{maker}\', {model}, \'{product_type}\')')

    file.write(',\n'.join(product_values) + ';\n')

# Создание и заполнение таблицы PC
with open('create_and_fill_PC.sql', 'w') as file:
    file.write(
        'CREATE TABLE PC (code INTEGER, model INTEGER, speed INTEGER, ram INTEGER, '
        'hd NUMERIC(10,1), cd VARCHAR(50), price NUMERIC(10,4));\n'
    )
    file.write('INSERT INTO PC\n')
    file.write('VALUES\n')

    pc_values = []
    for code, i in enumerate(range(1121, 1225), start=1):
        pc_values.append(
            f'({code}, {i}, {random.randrange(500, 901, 100)},'
            f' {random.randrange(32, 129, 32)}, {random.uniform(5.0, 21.0)},'
            f' \'{random.randrange(12, 53, 4)}x\', {random.uniform(350.0, 1001.0)})'
        )

    file.write(',\n'.join(pc_values) + ';\n')

# Создание и заполнение таблицы Laptop
with open('create_and_fill_Laptop.sql', 'w') as file:
    file.write(
        'CREATE TABLE Laptop (code INTEGER, model INTEGER, speed INTEGER, ram INTEGER, '
        'hd NUMERIC(10,1), screen NUMERIC(5,1), price NUMERIC(10,4));\n'
    )
    file.write('INSERT INTO Laptop\n')
    file.write('VALUES\n')

    laptop_values = []
    for code, i in enumerate(range(1, 101), start=1):
        laptop_values.append(
            f'({code}, {i}, {random.randrange(350, 1501, 100)},'
            f' {random.randrange(32, 129, 32)}, {random.uniform(5.0, 21.0)},'
            f' {random.uniform(10.0, 17.0)}, {random.uniform(350.0, 1001.0)})'
        )

    file.write(',\n'.join(laptop_values) + ';\n')

# Создание и заполнение таблицы Printer
with open('create_and_fill_Printer.sql', 'w') as file:
    file.write(
        'CREATE TABLE Printer (code INTEGER, model INTEGER, color CHAR(1), '
        'type VARCHAR(10), price NUMERIC(10,4));\n'
    )
    file.write('INSERT INTO Printer\n')
    file.write('VALUES\n')

    printer_values = []
    for code, i in enumerate(range(1, 101), start=1):
        color = random.choice(['y', 'n'])
        printer_type = random.choice(['Laser', 'Jet', 'Matrix'])
        printer_values.append(
            f'({code}, {i}, \'{color}\', \'{printer_type}\', {random.uniform(150.0, 601.0)})'
        )

    file.write(',\n'.join(printer_values) + ';\n')
