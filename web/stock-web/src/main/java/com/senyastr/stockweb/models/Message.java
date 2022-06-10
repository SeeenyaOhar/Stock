package com.senyastr.stockweb.models;

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;

@Entity
@Data
@Inheritance(strategy=InheritanceType.TABLE_PER_CLASS)
@NoArgsConstructor(access= AccessLevel.PUBLIC, force=true)
public class Message {
    @Id
    @GeneratedValue(strategy = GenerationType.TABLE)
    int id;
    String message;
    @Enumerated(EnumType.ORDINAL)
    Sender sender;
    public Message(int id, String message, Sender sender){
        this.id = id;
        this.message = message;
        this.sender = sender;
    }
}
