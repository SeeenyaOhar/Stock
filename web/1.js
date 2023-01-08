/**
 * @param {string} num1
 * @param {string} num2
 * @return {string}
 */
var addStrings = function(num1, num2) {
    if (num1.length != num2.length){
        let appendix = "";

        for (let i = 0; i < Math.abs(num1.length - num2.length); i++){
            appendix += "0";
        }

        if (num1.length > num2.length) num2 = appendix + num2;
        else num1 = appendix + num1;
    }  
    console.log(num1);
    console.log(num2);
    let result = "";
    let i = num2.length;
    let remainder = 0;
    while(i--){
        let a = num1.charCodeAt(i) - 48;
        let b = num2.charCodeAt(i) - 48;
        
        console.log(a, b);
        result += (a+b+remainder) % 10;

        remainder = ((a+b+remainder) >= 10) ? Math.floor((a+b+remainder) / 10) : 0; 
        
    }
    if (remainder != 0){
        result += remainder.toString();
    }
    let reversed = "";
    let j = result.length;
    while(j >= 0){
        reversed += result.charAt(j);
        j--;
    }

    return reversed;
};

console.log(addStrings("1", "9"))