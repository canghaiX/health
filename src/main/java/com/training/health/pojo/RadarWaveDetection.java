package com.training.health.pojo;

import lombok.*;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import java.util.Date;

@Getter
@Setter
@Data
@NoArgsConstructor
@EqualsAndHashCode
@Entity
@Table( name ="radar_wave_detection" )
public class RadarWaveDetection {
    @Id
    private String detectionId;
    @Column(name = "patient_id")
    private String patientId;
    @Column(name = "detection_date")
    private Date dectionDate;
    @Column(name = "detection_data")
    private String detectionData;
}
