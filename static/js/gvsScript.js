$(document).ready(function(){

//used to send a request in backqround can remove it if not needed
// 	console.log("loaded");
// 	$('a#launch').bind('click', function() {
//         $.getJSON('/camscan',
//             function(data) {
//           return
//         });
//         return false;
//       });
//used to send a request in backqround can remove it if not needed

	$('#printInvoice').click(function(){
	        Popup($('.invoice')[0].outerHTML);
	        function Popup(data)
	        {
	            window.print();
	            return true;
	        }
    });

    //Check File API support in billingPage
    if(window.File && window.FileList && window.FileReader)
    {
        $('body').on("change",'#files', function(event) {
            var files = event.target.files; //FileList object
            var output = document.getElementById("result");
            for(var i = 0; i< files.length; i++)
            {
                var file = files[i];
                //Only pics
                // if(!file.type.match('image'))
                if(file.type.match('image.*')){
                    if(this.files[0].size < 2097152){
                  // continue;
                    var picReader = new FileReader();
                    picReader.addEventListener("load",function(event){
                        var picFile = event.target;
                        var div = document.createElement("div");
                        div.innerHTML = "<img class='thumbnail' src='" + picFile.result + "'" +
                                "title='preview image'/>";
                        output.insertBefore(div,null);
                    });
                    //Read the image
                    $('#clear, #result').show();
                    picReader.readAsDataURL(file);
                    }else{
                        alert("Image Size is too big. Minimum size is 2MB.");
                        $(this).val("");
                    }
                }else{
                alert("You can only upload image file.");
                $(this).val("");
            }
            }

        });
    }
    else
    {
        console.log("Your browser does not support File API");
    }

  /*  $('body').on("click",'#files', function() {
        $('.thumbnail').parent().remove();
        $('result').hide();
        $(this).val("");
    }); */

    $('body').on("click",'#clear', function() {
        $('.thumbnail').parent().remove();
        $('#result').hide();
        $('#files').val("");
        $(this).hide();
    });
    //Check File API support in billingPage

});

