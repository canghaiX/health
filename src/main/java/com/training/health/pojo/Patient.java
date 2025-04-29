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
@Table( name ="patient" )
public class Patient {
    @Id
    private String patientId;
    private String name;
    private int age;
    private char gender;
    @Column(name = "contact_info")
    private String contactInfo;
    @Column(name = "id_number")
    private String idNumber;

}
