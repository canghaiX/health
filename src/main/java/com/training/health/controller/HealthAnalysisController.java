package com.training.health.controller;

import com.training.health.service.HealthAnalysisService;
import com.training.health.util.CommonResult;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;
import java.util.Map;

@RestController
public class HealthAnalysisController {
    @Resource
    HealthAnalysisService healthAnalysisService;
    @PostMapping("/health/show")
    public CommonResult showAll(HttpServletRequest request, @RequestBody Map<String,Object> reqs){
        return CommonResult.success("成功");
    }
}
