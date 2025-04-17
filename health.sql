/*
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 80021
 Source Host           : localhost:3306
 Source Schema         : health

 Target Server Type    : MySQL
 Target Server Version : 80021
 File Encoding         : 65001

 Date: 17/04/2025 17:40:19
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for health_analysis
-- ----------------------------
DROP TABLE IF EXISTS `health_analysis`;
CREATE TABLE `health_analysis`  (
  `medical_report` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '健康分析结果唯一标识，自增主键',
  `report_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关联体检报告表的报告 ID',
  `system_type` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '分析对应的系统类型（如呼吸系统、消化系统等）',
  `analysis_result` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '该系统的分析结果（如评估结论、建议等）',
  PRIMARY KEY (`medical_report`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for medical_report
-- ----------------------------
DROP TABLE IF EXISTS `medical_report`;
CREATE TABLE `medical_report`  (
  `reprot_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '体检报告唯一标识',
  `patient_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '外键，连接patient表',
  `report_date` datetime NOT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '体检报告日期',
  `institution` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '体检机构名称',
  `report_file` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '体检报告路径',
  `respiratory_data` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '呼吸系统体检数据（以 JSON 格式存储相关指标数据）',
  `digestive_data` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '消化系统体检数据（以 JSON 格式存储相关指标数据）',
  `immune_data` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '免疫系统体检数据（以 JSON 格式存储相关指标数据）',
  `endocrine_data` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '内分泌系统体检数据（以 JSON 格式存储相关指标数据）（按照体检报告样例创建，还有其他的系统数据）',
  PRIMARY KEY (`reprot_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for patient
-- ----------------------------
DROP TABLE IF EXISTS `patient`;
CREATE TABLE `patient`  (
  `patient_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '病人唯一标识',
  `name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '病人姓名',
  `age` int NOT NULL COMMENT '病人年龄',
  `gender` char(1) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '病人(m/f)',
  `contact_info` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '病人联系方式',
  `id_number` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '外键，连接注册表',
  PRIMARY KEY (`patient_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Table structure for radar_wave_detection
-- ----------------------------
DROP TABLE IF EXISTS `radar_wave_detection`;
CREATE TABLE `radar_wave_detection`  (
  `detection_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '雷达波检测记录唯一标识',
  `patient_id` varchar(18) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '关联病人表的病人 ID',
  `detection_date` datetime NULL DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP COMMENT '雷达波检测日期',
  `detection_data` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '雷达波检测的身体监测数据（可根据实际情况以合适格式存储，如 JSON 等）',
  PRIMARY KEY (`detection_id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
