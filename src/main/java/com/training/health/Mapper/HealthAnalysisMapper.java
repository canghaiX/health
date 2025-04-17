package com.training.health.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.training.health.pojo.HealthAnalysis;
import org.apache.ibatis.annotations.Mapper;
import org.springframework.stereotype.Repository;

@Repository
@Mapper
public interface HealthAnalysisMapper extends BaseMapper<HealthAnalysis> {
}
