package com.training.health.pojo;

import lombok.*;

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
}
