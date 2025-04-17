package com.training.health.service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.training.health.Mapper.HealthAnalysisMapper;
import com.training.health.Mapper.MedicalReportMapper;
import com.training.health.pojo.HealthAnalysis;
import com.training.health.pojo.MedicalReport;
import com.training.health.service.HealthAnalysisService;
import com.training.health.service.MedicalReportService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MedicalReportServiceImpl extends ServiceImpl<MedicalReportMapper, MedicalReport>implements MedicalReportService {
@Autowired
    MedicalReportMapper medicalReportMapper;
}
