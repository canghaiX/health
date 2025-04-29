package com.training.health.pojo;

import lombok.*;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import javax.xml.soap.Text;
import java.util.Date;

@Getter
@Setter
@Data
@NoArgsConstructor
@EqualsAndHashCode
@Entity
@Table( name ="medical_report" )
public class MedicalReport {
    @Id
    private String reportId;
    @Column(name = "patient_id")
    private String patientId;
    @Column(name = "report_date")
    private Date reportDate;
    private String institution;
    @Column(name="report_file")
    private String reportFile;
    @Column(name="respiratory_data")
    private String respiratoryData;
    @Column(name = "digestive_data")
    private String disgestiveData;
    @Column(name = "immune_data")
    private String immuneData;
    @Column(name="endocrine_data")
    private String encdocrineData;
}
