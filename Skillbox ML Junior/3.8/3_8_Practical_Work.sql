SET NAMES utf8;
SET time_zone = '+00:00';
SET foreign_key_checks = 0;
SET sql_mode = 'NO_AUTO_VALUE_ON_ZERO';

SET NAMES utf8mb4;

DROP TABLE IF EXISTS `Laptop`;
CREATE TABLE `Laptop` (
  `code` int DEFAULT NULL,
  `model` int DEFAULT NULL,
  `speed` int DEFAULT NULL,
  `ram` int DEFAULT NULL,
  `hd` decimal(10,1) DEFAULT NULL,
  `screen` decimal(5,2) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `Laptop` (`code`, `model`, `speed`, `ram`, `hd`, `screen`, `price`) VALUES
(1,	1,	950,	64,	9.9,	15.38,	540.71),
(2,	2,	950,	128,	15.2,	14.69,	913.76),
(3,	3,	950,	128,	16.4,	12.03,	484.79),
(4,	4,	550,	32,	7.9,	16.27,	569.72),
(5,	5,	950,	128,	17.7,	13.69,	379.60),
(6,	6,	1450,	64,	11.5,	12.49,	955.15),
(7,	7,	1450,	32,	6.3,	12.02,	909.12),
(8,	8,	350,	128,	5.5,	15.69,	510.75),
(9,	9,	950,	96,	19.1,	14.94,	790.02),
(10,	10,	650,	32,	20.6,	10.25,	944.23),
(11,	11,	350,	128,	19.6,	16.69,	497.09),
(12,	12,	350,	64,	13.8,	12.88,	840.18),
(13,	13,	1150,	128,	20.9,	14.00,	388.70),
(14,	14,	1150,	96,	8.9,	10.47,	927.72),
(15,	15,	450,	128,	11.5,	10.40,	840.57),
(16,	16,	350,	32,	11.6,	10.23,	560.81),
(17,	17,	1250,	96,	6.0,	12.90,	919.82),
(18,	18,	750,	96,	18.0,	13.31,	753.35),
(19,	19,	1450,	64,	13.6,	13.80,	630.83),
(20,	20,	650,	32,	6.5,	11.68,	412.71),
(21,	21,	1450,	64,	14.2,	14.62,	455.69),
(22,	22,	350,	96,	14.6,	12.58,	707.05),
(23,	23,	350,	128,	6.5,	12.43,	985.01),
(24,	24,	450,	128,	11.9,	11.42,	749.49),
(25,	25,	1250,	96,	11.0,	15.95,	441.27),
(26,	26,	1150,	32,	19.0,	14.05,	823.35),
(27,	27,	950,	64,	16.8,	12.00,	398.72),
(28,	28,	950,	64,	10.2,	11.77,	689.64),
(29,	29,	1050,	64,	12.8,	15.91,	551.82),
(30,	30,	1050,	64,	12.3,	16.33,	745.78),
(31,	31,	950,	64,	9.5,	15.99,	506.46),
(32,	32,	1450,	128,	12.5,	11.31,	535.54),
(33,	33,	950,	128,	19.9,	15.64,	353.66),
(34,	34,	850,	128,	9.5,	10.64,	839.10),
(35,	35,	1250,	128,	12.4,	10.38,	447.71),
(36,	36,	1350,	96,	6.8,	13.65,	522.51),
(37,	37,	550,	128,	13.7,	14.79,	805.76),
(38,	38,	1150,	96,	17.9,	16.43,	613.21),
(39,	39,	750,	64,	13.6,	10.59,	441.29),
(40,	40,	650,	96,	7.0,	14.43,	774.35),
(41,	41,	950,	32,	7.7,	11.40,	483.45),
(42,	42,	1350,	96,	20.7,	13.13,	484.97),
(43,	43,	950,	128,	10.8,	11.91,	853.65),
(44,	44,	1150,	96,	6.6,	16.00,	710.49),
(45,	45,	1050,	128,	16.3,	11.09,	872.89),
(46,	46,	850,	128,	17.5,	13.11,	556.15),
(47,	47,	550,	128,	16.7,	16.54,	367.32),
(48,	48,	750,	128,	7.5,	13.16,	769.84),
(49,	49,	450,	64,	12.3,	13.11,	371.13),
(50,	50,	950,	128,	11.3,	16.63,	649.23),
(51,	51,	350,	96,	5.7,	11.10,	524.06),
(52,	52,	1150,	32,	18.3,	15.17,	791.63),
(53,	53,	450,	64,	8.9,	11.63,	926.21),
(54,	54,	350,	96,	13.0,	11.88,	959.82),
(55,	55,	450,	64,	20.9,	10.50,	482.21),
(56,	56,	750,	96,	14.4,	16.65,	937.69),
(57,	57,	950,	64,	9.3,	14.88,	492.04),
(58,	58,	750,	64,	20.1,	11.26,	577.48),
(59,	59,	350,	96,	9.4,	14.81,	944.93),
(60,	60,	950,	64,	9.1,	10.84,	797.92),
(61,	61,	950,	32,	20.1,	11.43,	692.39),
(62,	62,	550,	96,	12.1,	12.06,	396.13),
(63,	63,	1350,	32,	9.5,	16.92,	772.35),
(64,	64,	1050,	32,	11.4,	15.99,	791.20),
(65,	65,	850,	128,	15.6,	16.87,	432.10),
(66,	66,	350,	64,	10.7,	13.36,	805.55),
(67,	67,	350,	32,	13.8,	11.88,	749.58),
(68,	68,	750,	64,	10.9,	10.90,	834.79),
(69,	69,	1350,	32,	15.9,	13.26,	686.82),
(70,	70,	750,	96,	5.2,	10.20,	450.12),
(71,	71,	350,	64,	8.4,	11.60,	887.84),
(72,	72,	450,	96,	11.0,	14.34,	541.36),
(73,	73,	850,	128,	9.1,	11.36,	836.74),
(74,	74,	350,	32,	7.5,	11.60,	932.88),
(75,	75,	850,	32,	20.5,	15.48,	679.25),
(76,	76,	350,	32,	14.4,	14.61,	765.90),
(77,	77,	350,	96,	6.9,	12.26,	364.73),
(78,	78,	1250,	96,	5.3,	11.60,	359.44),
(79,	79,	1150,	96,	8.6,	13.24,	355.67),
(80,	80,	850,	96,	17.5,	14.88,	708.77),
(81,	81,	550,	96,	11.8,	14.63,	881.25),
(82,	82,	750,	128,	18.9,	11.92,	738.19),
(83,	83,	1050,	64,	18.4,	13.36,	858.86),
(84,	84,	550,	96,	19.9,	16.60,	482.36),
(85,	85,	1450,	32,	12.1,	13.43,	765.73),
(86,	86,	550,	32,	12.8,	16.40,	463.42),
(87,	87,	1150,	32,	14.0,	10.10,	501.89),
(88,	88,	350,	64,	20.0,	14.34,	442.53),
(89,	89,	450,	96,	8.4,	14.57,	498.16),
(90,	90,	1250,	32,	15.5,	11.54,	466.12),
(91,	91,	350,	96,	6.2,	13.94,	824.90),
(92,	92,	550,	64,	5.1,	13.36,	658.74),
(93,	93,	350,	128,	20.0,	14.44,	415.19),
(94,	94,	550,	96,	12.4,	11.69,	664.14),
(95,	95,	1150,	64,	13.2,	14.60,	749.48),
(96,	96,	1350,	96,	17.8,	15.42,	908.01),
(97,	97,	1150,	64,	15.0,	16.95,	630.59),
(98,	98,	750,	128,	17.3,	11.07,	861.77),
(99,	99,	350,	64,	20.7,	16.17,	806.30),
(100,	100,	1450,	96,	19.5,	15.41,	701.76),
(1,	1,	950,	64,	9.9,	15.38,	540.71),
(2,	2,	950,	128,	15.2,	14.69,	913.76),
(3,	3,	950,	128,	16.4,	12.03,	484.79),
(4,	4,	550,	32,	7.9,	16.27,	569.72),
(5,	5,	950,	128,	17.7,	13.69,	379.60),
(6,	6,	1450,	64,	11.5,	12.49,	955.15),
(7,	7,	1450,	32,	6.3,	12.02,	909.12),
(8,	8,	350,	128,	5.5,	15.69,	510.75),
(9,	9,	950,	96,	19.1,	14.94,	790.02),
(10,	10,	650,	32,	20.6,	10.25,	944.23),
(11,	11,	350,	128,	19.6,	16.69,	497.09),
(12,	12,	350,	64,	13.8,	12.88,	840.18),
(13,	13,	1150,	128,	20.9,	14.00,	388.70),
(14,	14,	1150,	96,	8.9,	10.47,	927.72),
(15,	15,	450,	128,	11.5,	10.40,	840.57),
(16,	16,	350,	32,	11.6,	10.23,	560.81),
(17,	17,	1250,	96,	6.0,	12.90,	919.82),
(18,	18,	750,	96,	18.0,	13.31,	753.35),
(19,	19,	1450,	64,	13.6,	13.80,	630.83),
(20,	20,	650,	32,	6.5,	11.68,	412.71),
(21,	21,	1450,	64,	14.2,	14.62,	455.69),
(22,	22,	350,	96,	14.6,	12.58,	707.05),
(23,	23,	350,	128,	6.5,	12.43,	985.01),
(24,	24,	450,	128,	11.9,	11.42,	749.49),
(25,	25,	1250,	96,	11.0,	15.95,	441.27),
(26,	26,	1150,	32,	19.0,	14.05,	823.35),
(27,	27,	950,	64,	16.8,	12.00,	398.72),
(28,	28,	950,	64,	10.2,	11.77,	689.64),
(29,	29,	1050,	64,	12.8,	15.91,	551.82),
(30,	30,	1050,	64,	12.3,	16.33,	745.78),
(31,	31,	950,	64,	9.5,	15.99,	506.46),
(32,	32,	1450,	128,	12.5,	11.31,	535.54),
(33,	33,	950,	128,	19.9,	15.64,	353.66),
(34,	34,	850,	128,	9.5,	10.64,	839.10),
(35,	35,	1250,	128,	12.4,	10.38,	447.71),
(36,	36,	1350,	96,	6.8,	13.65,	522.51),
(37,	37,	550,	128,	13.7,	14.79,	805.76),
(38,	38,	1150,	96,	17.9,	16.43,	613.21),
(39,	39,	750,	64,	13.6,	10.59,	441.29),
(40,	40,	650,	96,	7.0,	14.43,	774.35),
(41,	41,	950,	32,	7.7,	11.40,	483.45),
(42,	42,	1350,	96,	20.7,	13.13,	484.97),
(43,	43,	950,	128,	10.8,	11.91,	853.65),
(44,	44,	1150,	96,	6.6,	16.00,	710.49),
(45,	45,	1050,	128,	16.3,	11.09,	872.89),
(46,	46,	850,	128,	17.5,	13.11,	556.15),
(47,	47,	550,	128,	16.7,	16.54,	367.32),
(48,	48,	750,	128,	7.5,	13.16,	769.84),
(49,	49,	450,	64,	12.3,	13.11,	371.13),
(50,	50,	950,	128,	11.3,	16.63,	649.23),
(51,	51,	350,	96,	5.7,	11.10,	524.06),
(52,	52,	1150,	32,	18.3,	15.17,	791.63),
(53,	53,	450,	64,	8.9,	11.63,	926.21),
(54,	54,	350,	96,	13.0,	11.88,	959.82),
(55,	55,	450,	64,	20.9,	10.50,	482.21),
(56,	56,	750,	96,	14.4,	16.65,	937.69),
(57,	57,	950,	64,	9.3,	14.88,	492.04),
(58,	58,	750,	64,	20.1,	11.26,	577.48),
(59,	59,	350,	96,	9.4,	14.81,	944.93),
(60,	60,	950,	64,	9.1,	10.84,	797.92),
(61,	61,	950,	32,	20.1,	11.43,	692.39),
(62,	62,	550,	96,	12.1,	12.06,	396.13),
(63,	63,	1350,	32,	9.5,	16.92,	772.35),
(64,	64,	1050,	32,	11.4,	15.99,	791.20),
(65,	65,	850,	128,	15.6,	16.87,	432.10),
(66,	66,	350,	64,	10.7,	13.36,	805.55),
(67,	67,	350,	32,	13.8,	11.88,	749.58),
(68,	68,	750,	64,	10.9,	10.90,	834.79),
(69,	69,	1350,	32,	15.9,	13.26,	686.82),
(70,	70,	750,	96,	5.2,	10.20,	450.12),
(71,	71,	350,	64,	8.4,	11.60,	887.84),
(72,	72,	450,	96,	11.0,	14.34,	541.36),
(73,	73,	850,	128,	9.1,	11.36,	836.74),
(74,	74,	350,	32,	7.5,	11.60,	932.88),
(75,	75,	850,	32,	20.5,	15.48,	679.25),
(76,	76,	350,	32,	14.4,	14.61,	765.90),
(77,	77,	350,	96,	6.9,	12.26,	364.73),
(78,	78,	1250,	96,	5.3,	11.60,	359.44),
(79,	79,	1150,	96,	8.6,	13.24,	355.67),
(80,	80,	850,	96,	17.5,	14.88,	708.77),
(81,	81,	550,	96,	11.8,	14.63,	881.25),
(82,	82,	750,	128,	18.9,	11.92,	738.19),
(83,	83,	1050,	64,	18.4,	13.36,	858.86),
(84,	84,	550,	96,	19.9,	16.60,	482.36),
(85,	85,	1450,	32,	12.1,	13.43,	765.73),
(86,	86,	550,	32,	12.8,	16.40,	463.42),
(87,	87,	1150,	32,	14.0,	10.10,	501.89),
(88,	88,	350,	64,	20.0,	14.34,	442.53),
(89,	89,	450,	96,	8.4,	14.57,	498.16),
(90,	90,	1250,	32,	15.5,	11.54,	466.12),
(91,	91,	350,	96,	6.2,	13.94,	824.90),
(92,	92,	550,	64,	5.1,	13.36,	658.74),
(93,	93,	350,	128,	20.0,	14.44,	415.19),
(94,	94,	550,	96,	12.4,	11.69,	664.14),
(95,	95,	1150,	64,	13.2,	14.60,	749.48),
(96,	96,	1350,	96,	17.8,	15.42,	908.01),
(97,	97,	1150,	64,	15.0,	16.95,	630.59),
(98,	98,	750,	128,	17.3,	11.07,	861.77),
(99,	99,	350,	64,	20.7,	16.17,	806.30),
(100,	100,	1450,	96,	19.5,	15.41,	701.76),
(13,	1321,	500,	64,	8.0,	12.00,	970.00);

DROP TABLE IF EXISTS `PC`;
CREATE TABLE `PC` (
  `code` int DEFAULT NULL,
  `model` int DEFAULT NULL,
  `speed` int DEFAULT NULL,
  `ram` int DEFAULT NULL,
  `hd` decimal(10,1) DEFAULT NULL,
  `cd` varchar(50) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `PC` (`code`, `model`, `speed`, `ram`, `hd`, `cd`, `price`) VALUES
(1,	1121,	500,	32,	12.9,	'12x',	355.37),
(2,	1122,	600,	64,	17.1,	'28x',	990.59),
(3,	1123,	700,	96,	20.9,	'36x',	647.37),
(4,	1124,	800,	96,	19.0,	'20x',	973.72),
(5,	1125,	900,	96,	12.2,	'16x',	549.29),
(6,	1126,	800,	64,	12.9,	'52x',	717.65),
(7,	1127,	900,	128,	16.4,	'16x',	722.36),
(8,	1128,	500,	96,	16.6,	'40x',	752.43),
(9,	1129,	600,	96,	8.4,	'44x',	397.24),
(10,	1130,	900,	96,	10.8,	'40x',	468.94),
(11,	1131,	900,	96,	14.4,	'12x',	937.15),
(12,	1132,	900,	128,	8.0,	'40x',	668.07),
(13,	1133,	700,	64,	16.0,	'24x',	918.72),
(14,	1134,	600,	128,	5.8,	'24x',	389.96),
(15,	1135,	800,	32,	12.4,	'52x',	977.88),
(16,	1136,	600,	64,	5.5,	'36x',	370.23),
(17,	1137,	500,	64,	12.0,	'36x',	660.36),
(18,	1138,	800,	96,	17.0,	'44x',	1000.77),
(19,	1139,	900,	128,	6.0,	'48x',	603.36),
(20,	1140,	700,	128,	14.7,	'44x',	805.21),
(21,	1141,	800,	128,	16.6,	'12x',	384.09),
(22,	1142,	800,	64,	7.2,	'16x',	383.17),
(23,	1143,	900,	64,	5.9,	'52x',	467.77),
(24,	1144,	800,	32,	16.9,	'20x',	743.89),
(25,	1145,	600,	32,	20.4,	'36x',	779.68),
(26,	1146,	900,	128,	9.8,	'44x',	786.98),
(27,	1147,	500,	32,	15.8,	'20x',	914.94),
(28,	1148,	900,	32,	18.6,	'48x',	896.38),
(29,	1149,	900,	128,	15.3,	'12x',	418.77),
(30,	1150,	600,	96,	15.8,	'48x',	572.59),
(31,	1151,	500,	96,	6.0,	'40x',	903.75),
(32,	1152,	900,	96,	6.4,	'32x',	829.29),
(33,	1153,	800,	32,	20.3,	'52x',	986.86),
(34,	1154,	500,	64,	5.1,	'48x',	884.85),
(35,	1155,	900,	32,	8.9,	'12x',	570.13),
(36,	1156,	500,	128,	16.7,	'20x',	455.73),
(37,	1157,	900,	64,	15.9,	'44x',	351.42),
(38,	1158,	700,	32,	6.0,	'16x',	809.28),
(39,	1159,	700,	96,	6.0,	'36x',	813.49),
(40,	1160,	900,	32,	16.2,	'28x',	842.30),
(41,	1161,	500,	32,	16.3,	'24x',	882.60),
(42,	1162,	900,	128,	9.4,	'24x',	796.12),
(43,	1163,	800,	64,	18.5,	'44x',	882.59),
(44,	1164,	800,	64,	14.3,	'16x',	493.30),
(45,	1165,	500,	64,	6.5,	'24x',	833.20),
(46,	1166,	900,	96,	14.0,	'32x',	787.14),
(47,	1167,	800,	96,	8.4,	'20x',	581.68),
(48,	1168,	800,	128,	20.0,	'48x',	515.01),
(49,	1169,	700,	64,	6.8,	'40x',	551.00),
(50,	1170,	500,	128,	19.1,	'52x',	852.50),
(51,	1171,	700,	96,	15.2,	'44x',	685.90),
(52,	1172,	500,	128,	16.8,	'44x',	599.06),
(53,	1173,	500,	32,	16.8,	'12x',	966.79),
(54,	1174,	500,	32,	17.6,	'20x',	413.24),
(55,	1175,	800,	96,	11.2,	'32x',	720.35),
(56,	1176,	700,	32,	16.2,	'12x',	887.32),
(57,	1177,	800,	64,	7.3,	'28x',	448.18),
(58,	1178,	700,	32,	18.3,	'12x',	494.27),
(59,	1179,	500,	96,	18.1,	'28x',	924.79),
(60,	1180,	700,	96,	7.5,	'24x',	394.71),
(61,	1181,	600,	32,	13.0,	'20x',	510.88),
(62,	1182,	500,	128,	17.2,	'24x',	888.14),
(63,	1183,	700,	64,	16.8,	'52x',	449.34),
(64,	1184,	900,	64,	5.2,	'32x',	864.69),
(65,	1185,	500,	32,	13.4,	'28x',	640.62),
(66,	1186,	700,	96,	10.9,	'32x',	802.83),
(67,	1187,	700,	32,	11.5,	'44x',	575.72),
(68,	1188,	900,	64,	20.5,	'16x',	620.93),
(69,	1189,	600,	32,	13.2,	'44x',	742.71),
(70,	1190,	700,	96,	6.1,	'12x',	491.19),
(71,	1191,	800,	32,	18.9,	'28x',	392.03),
(72,	1192,	700,	32,	6.8,	'52x',	757.40),
(73,	1193,	900,	32,	20.7,	'20x',	362.20),
(74,	1194,	900,	128,	20.4,	'16x',	887.53),
(75,	1195,	700,	64,	13.4,	'36x',	914.86),
(76,	1196,	500,	96,	6.9,	'20x',	676.47),
(77,	1197,	800,	64,	8.2,	'24x',	501.52),
(78,	1198,	800,	64,	10.8,	'44x',	760.99),
(79,	1199,	600,	32,	17.5,	'36x',	551.81),
(80,	1200,	600,	64,	8.1,	'36x',	397.39),
(81,	1201,	800,	128,	14.4,	'48x',	541.44),
(82,	1202,	600,	32,	16.5,	'52x',	914.82),
(83,	1203,	800,	128,	18.9,	'48x',	983.63),
(84,	1204,	600,	64,	13.4,	'16x',	363.72),
(85,	1205,	600,	96,	7.3,	'32x',	722.57),
(86,	1206,	500,	32,	11.8,	'28x',	460.54),
(87,	1207,	500,	32,	9.3,	'16x',	395.66),
(88,	1208,	800,	64,	5.6,	'20x',	564.86),
(89,	1209,	800,	128,	9.8,	'36x',	451.95),
(90,	1210,	500,	96,	9.6,	'32x',	368.32),
(91,	1211,	800,	32,	14.7,	'40x',	824.33),
(92,	1212,	800,	64,	12.8,	'40x',	985.69),
(93,	1213,	700,	128,	10.7,	'12x',	865.10),
(94,	1214,	900,	64,	15.3,	'40x',	659.81),
(95,	1215,	700,	128,	14.5,	'12x',	632.63),
(96,	1216,	900,	128,	7.4,	'40x',	399.24),
(97,	1217,	700,	64,	9.2,	'32x',	982.34),
(98,	1218,	500,	128,	14.3,	'52x',	827.45),
(99,	1219,	500,	64,	12.0,	'28x',	793.21),
(100,	1220,	700,	128,	19.3,	'20x',	813.41),
(101,	1221,	600,	128,	20.8,	'16x',	619.75),
(102,	1222,	900,	96,	11.1,	'32x',	549.38),
(103,	1223,	900,	96,	12.6,	'20x',	455.79),
(104,	1224,	900,	96,	5.8,	'36x',	543.52),
(1226,	1226,	500,	64,	5.0,	'12x',	600.00);

DROP TABLE IF EXISTS `Printer`;
CREATE TABLE `Printer` (
  `code` int DEFAULT NULL,
  `model` int DEFAULT NULL,
  `color_type` varchar(1) DEFAULT NULL,
  `type` varchar(255) DEFAULT NULL,
  `price` decimal(10,2) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `Printer` (`code`, `model`, `color_type`, `type`, `price`) VALUES
(1,	1,	'y',	'Jet',	577.43),
(2,	2,	'n',	'Jet',	599.62),
(3,	3,	'n',	'Matrix',	222.17),
(4,	4,	'n',	'Matrix',	239.76),
(5,	5,	'n',	'Matrix',	240.88),
(6,	6,	'n',	'Jet',	243.66),
(7,	7,	'y',	'Laser',	197.16),
(8,	8,	'n',	'Jet',	189.67),
(9,	9,	'n',	'Matrix',	233.65),
(10,	10,	'n',	'Jet',	600.75),
(11,	11,	'n',	'Matrix',	341.17),
(12,	12,	'y',	'Jet',	336.83),
(13,	13,	'n',	'Jet',	458.21),
(14,	14,	'n',	'Laser',	332.66),
(15,	15,	'n',	'Matrix',	282.65),
(16,	16,	'n',	'Jet',	527.21),
(17,	17,	'y',	'Jet',	590.27),
(18,	18,	'n',	'Laser',	152.54),
(19,	19,	'y',	'Jet',	499.00),
(20,	20,	'n',	'Jet',	334.91),
(21,	21,	'y',	'Jet',	253.27),
(22,	22,	'n',	'Matrix',	166.56),
(23,	23,	'n',	'Jet',	503.74),
(24,	24,	'y',	'Laser',	334.78),
(25,	25,	'y',	'Laser',	184.21),
(26,	26,	'n',	'Laser',	315.80),
(27,	27,	'n',	'Matrix',	196.90),
(28,	28,	'n',	'Matrix',	581.19),
(29,	29,	'y',	'Laser',	525.97),
(30,	30,	'y',	'Jet',	416.25),
(31,	31,	'n',	'Jet',	183.87),
(32,	32,	'y',	'Matrix',	215.43),
(33,	33,	'y',	'Laser',	220.88),
(34,	34,	'y',	'Matrix',	294.08),
(35,	35,	'n',	'Laser',	516.68),
(36,	36,	'n',	'Laser',	392.55),
(37,	37,	'n',	'Laser',	421.00),
(38,	38,	'n',	'Matrix',	517.15),
(39,	39,	'n',	'Laser',	590.18),
(40,	40,	'n',	'Jet',	386.74),
(41,	41,	'y',	'Jet',	405.36),
(42,	42,	'n',	'Matrix',	495.73),
(43,	43,	'y',	'Jet',	284.73),
(44,	44,	'n',	'Jet',	257.43),
(45,	45,	'n',	'Matrix',	386.56),
(46,	46,	'n',	'Laser',	472.39),
(47,	47,	'y',	'Matrix',	495.56),
(48,	48,	'y',	'Laser',	172.09),
(49,	49,	'y',	'Matrix',	403.91),
(50,	50,	'n',	'Matrix',	384.73),
(51,	51,	'n',	'Jet',	385.81),
(52,	52,	'n',	'Jet',	315.54),
(53,	53,	'y',	'Laser',	482.31),
(54,	54,	'n',	'Jet',	508.02),
(55,	55,	'n',	'Laser',	377.65),
(56,	56,	'n',	'Matrix',	534.91),
(57,	57,	'y',	'Jet',	558.15),
(58,	58,	'n',	'Matrix',	237.76),
(59,	59,	'n',	'Jet',	293.07),
(60,	60,	'y',	'Matrix',	399.95),
(61,	61,	'n',	'Laser',	576.22),
(62,	62,	'n',	'Laser',	307.58),
(63,	63,	'y',	'Jet',	177.67),
(64,	64,	'n',	'Laser',	214.24),
(65,	65,	'n',	'Matrix',	249.88),
(66,	66,	'n',	'Matrix',	374.76),
(67,	67,	'y',	'Matrix',	408.56),
(68,	68,	'n',	'Matrix',	600.01),
(69,	69,	'y',	'Laser',	379.31),
(70,	70,	'y',	'Jet',	354.73),
(71,	71,	'n',	'Laser',	477.82),
(72,	72,	'y',	'Laser',	307.35),
(73,	73,	'y',	'Jet',	315.17),
(74,	74,	'n',	'Matrix',	513.93),
(75,	75,	'y',	'Laser',	196.05),
(76,	76,	'n',	'Laser',	520.60),
(77,	77,	'y',	'Laser',	404.55),
(78,	78,	'y',	'Laser',	189.53),
(79,	79,	'y',	'Jet',	396.45),
(80,	80,	'y',	'Matrix',	384.96),
(81,	81,	'n',	'Matrix',	529.35),
(82,	82,	'n',	'Laser',	571.63),
(83,	83,	'y',	'Matrix',	504.20),
(84,	84,	'n',	'Laser',	556.47),
(85,	85,	'y',	'Matrix',	349.43),
(86,	86,	'n',	'Jet',	358.35),
(87,	87,	'n',	'Matrix',	462.26),
(88,	88,	'y',	'Jet',	206.83),
(89,	89,	'n',	'Matrix',	331.02),
(90,	90,	'n',	'Matrix',	272.09),
(91,	91,	'y',	'Laser',	428.63),
(92,	92,	'n',	'Jet',	344.99),
(93,	93,	'n',	'Jet',	360.76),
(94,	94,	'n',	'Laser',	412.64),
(95,	95,	'y',	'Matrix',	548.01),
(96,	96,	'n',	'Jet',	378.95),
(97,	97,	'y',	'Laser',	305.09),
(98,	98,	'y',	'Laser',	320.10),
(99,	99,	'n',	'Laser',	433.86),
(100,	100,	'y',	'Laser',	517.34);

DROP TABLE IF EXISTS `Product`;
CREATE TABLE `Product` (
  `maker` varchar(255) DEFAULT NULL,
  `model` int DEFAULT NULL,
  `type` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

INSERT INTO `Product` (`maker`, `model`, `type`) VALUES
('A',	8792,	'PC'),
('D',	5671,	'Laptop'),
('D',	5552,	'Laptop'),
('E',	6925,	'Printer'),
('B',	9307,	'Laptop'),
('D',	7145,	'Printer'),
('E',	5767,	'PC'),
('C',	2568,	'Laptop'),
('C',	6703,	'PC'),
('C',	1397,	'Laptop'),
('B',	4672,	'Printer'),
('C',	1594,	'Printer'),
('E',	9859,	'PC'),
('D',	4766,	'PC'),
('C',	8871,	'Printer'),
('B',	1575,	'Laptop'),
('B',	8448,	'Printer'),
('C',	5815,	'Printer'),
('C',	4184,	'Printer'),
('C',	3308,	'PC'),
('D',	5477,	'Laptop'),
('B',	8483,	'PC'),
('D',	4301,	'Laptop'),
('B',	8320,	'Printer'),
('E',	4598,	'PC'),
('B',	4769,	'Laptop'),
('A',	4608,	'PC'),
('E',	3037,	'PC'),
('A',	9020,	'Printer'),
('B',	5625,	'Laptop'),
('C',	5635,	'Printer'),
('B',	6663,	'Printer'),
('D',	3430,	'Laptop'),
('C',	6615,	'Laptop'),
('A',	7275,	'PC'),
('D',	2761,	'PC'),
('B',	9386,	'Laptop'),
('C',	8951,	'Printer'),
('C',	2081,	'Laptop'),
('A',	7937,	'Laptop'),
('E',	9758,	'Printer'),
('B',	1185,	'Laptop'),
('E',	7410,	'PC'),
('C',	7908,	'PC'),
('D',	4977,	'Laptop'),
('D',	7109,	'Laptop'),
('D',	2808,	'PC'),
('B',	6861,	'Laptop'),
('E',	3597,	'Laptop'),
('B',	9153,	'PC'),
('A',	6397,	'Printer'),
('E',	6284,	'Printer'),
('A',	6331,	'PC'),
('E',	1951,	'Printer'),
('B',	8654,	'PC'),
('E',	6386,	'PC'),
('A',	6171,	'Laptop'),
('B',	3732,	'Laptop'),
('A',	3845,	'Printer'),
('E',	9904,	'Printer'),
('E',	2478,	'Printer'),
('D',	6744,	'PC'),
('D',	2870,	'Laptop'),
('D',	6040,	'Laptop'),
('E',	6323,	'Printer'),
('B',	3252,	'Printer'),
('D',	6408,	'Printer'),
('A',	7350,	'Laptop'),
('B',	2469,	'Printer'),
('A',	9864,	'Laptop'),
('D',	3860,	'Printer'),
('B',	2147,	'Laptop'),
('C',	6850,	'Printer'),
('D',	6187,	'Laptop'),
('B',	6897,	'PC'),
('D',	6665,	'Laptop'),
('E',	3836,	'PC'),
('B',	8953,	'Printer'),
('E',	7304,	'Printer'),
('D',	9286,	'PC'),
('C',	5824,	'Laptop'),
('D',	2157,	'Laptop'),
('D',	9284,	'Printer'),
('C',	9738,	'PC'),
('C',	5260,	'PC'),
('D',	1538,	'Laptop'),
('A',	4455,	'Laptop'),
('E',	1724,	'Printer'),
('A',	1391,	'Printer'),
('B',	5357,	'PC'),
('A',	9733,	'Laptop'),
('C',	1322,	'PC'),
('A',	9096,	'Printer'),
('C',	7792,	'Laptop'),
('A',	9747,	'PC'),
('E',	1351,	'PC'),
('A',	4690,	'Printer'),
('C',	3003,	'Printer'),
('C',	6083,	'Printer'),
('D',	3038,	'Printer');

-- 2024-02-28 18:35:48