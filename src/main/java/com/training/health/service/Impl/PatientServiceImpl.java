package com.training.health.service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.training.health.Mapper.HealthAnalysisMapper;
import com.training.health.Mapper.PatientMapper;
import com.training.health.pojo.HealthAnalysis;
import com.training.health.pojo.Patient;
import com.training.health.service.HealthAnalysisService;
import com.training.health.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class PatientServiceImpl extends ServiceImpl<PatientMapper, Patient>implements PatientService {
    @Autowired
    PatientMapper patientMapper;
}
