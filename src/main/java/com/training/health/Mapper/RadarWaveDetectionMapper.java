package com.training.health.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.training.health.pojo.Patient;
import com.training.health.pojo.RadarWaveDetection;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

@Repository
@Mapper
public interface RadarWaveDetectionMapper extends BaseMapper<RadarWaveDetection> {
}
