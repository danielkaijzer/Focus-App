const Chart = require('chart.js/auto');
const annotationPlugin = require('chartjs-plugin-annotation');
const { ipcRenderer } = require('electron');
Chart.register(annotationPlugin);

// document.getElementById('myButton').addEventListener('click', () => {
//   // Send a message to the main process to open a new page.
//   ipcRenderer.send('open-new-page');
// });


const ctx = document.getElementById('gaugeChart').getContext('2d');
const score = 93;
const maxScore = 100;

new Chart(ctx, {
  type: 'doughnut',
  data: {
    datasets: [
      {
        data: [score, maxScore - score],
        backgroundColor: ['#3da8f5', '#e0e0e0'],
        borderWidth: 0,
      },
    ],
  },
  options: {
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
      annotation: {
        annotations: {
          acquiredScore: {
            type: 'label',
            xValue: 0.5,
            yValue: -1.1,
            backgroundColor: 'transparent',
            content: [`Productivity Score: ${score} / ${maxScore}`],
            color: 'white',
            font: {
              size: 14,
              weight: 'bold',
            },
          },


        },
      },
    },
    cutout: '80%',
    rotation: -90,
    circumference: 180,
  },
});
