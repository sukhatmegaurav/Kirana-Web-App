$(document).ready(function(){
	console.log("loaded");
	$('a#launch').bind('click', function() {
        $.getJSON('/camscan',
            function(data) {
          //do nothing
        });
        return false;
      });
});