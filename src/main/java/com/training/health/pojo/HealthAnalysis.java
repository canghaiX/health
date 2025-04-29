package com.training.health.pojo;

import lombok.*;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

@Getter
@Setter
@Data
@NoArgsConstructor
@EqualsAndHashCode
@Entity
@Table( name ="health_analysis" )
public class HealthAnalysis {
    @Id
    private String medical_report;
    @Column(name = "report_id")
    private String reportId;
    @Column(name = "system_type")
    private String systemType;
    @Column(name = "analysis_result")
    private String analysisResult;
}
