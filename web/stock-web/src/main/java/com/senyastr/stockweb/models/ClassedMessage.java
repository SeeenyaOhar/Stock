package com.senyastr.stockweb.models;

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.*;

@Entity()
@Table(name= "classed_messages")
@Data
@NoArgsConstructor(access= AccessLevel.PUBLIC, force=true)
public class ClassedMessage extends Message {
    @Enumerated(value= EnumType.ORDINAL)
    @Column(name="MESSAGE_CLASS")
    MessageClass messageClass;
    public ClassedMessage(int id, String message, Sender sender, MessageClass messageClass){
        super(id, message, sender);
        this.messageClass = messageClass;
    }

}
