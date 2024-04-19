-- Создание таблицы Product
CREATE TABLE Product (
    maker VARCHAR(255),
    model INTEGER,
    type VARCHAR(255)
);

-- Создание таблицы PC
CREATE TABLE PC (
    code INTEGER,
    model INTEGER,
    speed INTEGER,
    ram INTEGER,
    hd NUMERIC(10,1),
    cd VARCHAR(50),
    price NUMERIC(10,2)
);

-- Создание таблицы Laptop
CREATE TABLE Laptop (
    code INTEGER,
    model INTEGER,
    speed INTEGER,
    ram INTEGER,
    hd NUMERIC(10,1),
    screen NUMERIC(5,2),
    price NUMERIC(10,2)
);

-- Создание таблицы Printer
CREATE TABLE Printer (
    code INTEGER,
    model INTEGER,
    color VARCHAR(1),
    type VARCHAR(255),
    price NUMERIC(10,2)
);
