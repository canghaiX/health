package com.training.health.service.Impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.training.health.Mapper.HealthAnalysisMapper;
import com.training.health.Mapper.RadarWaveDetectionMapper;
import com.training.health.pojo.HealthAnalysis;
import com.training.health.pojo.RadarWaveDetection;
import com.training.health.service.HealthAnalysisService;
import com.training.health.service.RadarWaveDetectionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class RadarWaveDetectionServiceImpl extends ServiceImpl<RadarWaveDetectionMapper, RadarWaveDetection>implements RadarWaveDetectionService {
    @Autowired
    RadarWaveDetectionMapper radarWaveDetectionMapper;
}
