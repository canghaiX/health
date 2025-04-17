package com.training.health.service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.training.health.Mapper.HealthAnalysisMapper;
import com.training.health.pojo.HealthAnalysis;
import com.training.health.service.HealthAnalysisService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class HealthAnalysisServiceImpl extends ServiceImpl<HealthAnalysisMapper, HealthAnalysis>implements HealthAnalysisService {
    @Autowired
    HealthAnalysisMapper healthAnalysisMapper;
}
