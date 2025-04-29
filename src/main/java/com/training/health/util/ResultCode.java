package com.training.health.util;

import lombok.Getter;
import lombok.Setter;

@Getter


public enum ResultCode {
    SUCCESS(200,"操作成功"),
    FAILED(500,"操作失败"),
    NOTOKEN(401,"未登录或登录已超时"),
    NOPERMISS(403,"无操作权限"),
    NOHANDLER(404,"请求地址错误")
    ;

    private Integer code;
    private String message;

    ResultCode(Integer code, String message) {
        this.code = code;
        this.message = message;
    }
}
