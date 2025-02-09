const fs = require('fs');

function loadJsonSync(filename) {
  try {
    const data = fs.readFileSync(filename, "utf8"); // Read file synchronously
    const jsonData = JSON.parse(data); // Parse JSON
    let focusTimes = jsonData.map(item => item.focusTime); // Extract focusTime values
    console.log(focusTimes); // Output: [23, 42, 63, 55, 73, 75]
    return focusTimes;
  } catch (error) {
    console.error("Error:", error);
    return null;
  }
}
focusTimes = loadJsonSync("./data.json");

var chartOptions = {
  chart: {
    height: 400,
    type: 'line',
    fontFamily: 'Helvetica, Arial, sans-serif',
    foreColor: 'rgb(215, 215, 215)',
    toolbar: {
      show: false,
    },
  },
  stroke: {
    curve: 'smooth',
    width: 2,
  },
  series: [
    {
      name: 'Focused',
      data: focusTimes,
      color: "rgb(59, 241, 31)"
    },
    {
      name: 'Distracted',
      data: focusTimes.map(item => 100 - item),
      color: "rgb(98, 98, 98)"
    }
  ],
  title: {
    text: 'Progess',
    align: 'left',
    offsetY: 25,
    offsetX: 5,
    style: {
      fontSize: '14px',
      fontWeight: 'bold',
      color: '#373d3f',
    },
  },
  markers: {
    size: 6,
    strokeWidth: 0,
    hover: {
      size: 9,
    },
    colors: ["rgb(59, 241, 31)", "rgb(98, 98, 98)"]
  },
  grid: {
    show: true,
    padding: {
      bottom: 0,
    },
  },
  labels: ['1st', '2nd', '3rd', '4th', '5th', '6th'],
  xaxis: {
    tooltip: {
      enabled: false
    },
  },
  legend: {
    position: 'top',
    horizontalAlign: 'right',
    offsetY: -10,
    labels: {
      colors: '#ffffff',
    },
  },
  grid: {
    borderColor: '#424242',
    xaxis: {
      lines: {
        show: true,
      },
    },
  },
};

var lineChart = new ApexCharts(document.querySelector('#line-chart'), chartOptions);
lineChart.render();























































  
  
      
    
    
  


  
        
    
  


    
    
  
  
  


    

