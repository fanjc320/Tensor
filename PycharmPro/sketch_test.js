'use strict';


function preload() {
  if (window.location.protocol != "https:" && window.location.protocol != 'file:')
    window.location.href = "https:" + window.location.href.substring(window.location.protocol.length);
}

function setup() {
  if (!window.AudioContext) {
    window.alert(Locale.get('no_support'));
  }
  data.init();
  data.control = new Control();
  data.equal_loud = new EqualLoud();
  frameRate(data.fps);
  var cnv = createCanvas(data.width + data.padding_right, 800);
  cnv.parent('draw_pane');
  cnv.mouseClicked(data.control.onClick);
  data.fft = new p5.FFT(data.smooth, data.fft_size);
  if (!data.no_mic) {
    data.mic_fft = new p5.FFT(data.smooth, data.fft_size);
  }

  var x1 = 2;
  var x2 = data.width;
  var y1 = 1;
  var y2 = 0;
   
  if (data.show_vspec) {
    y2 = y1 + 450;
    vspec.init(x1, x2, y1, y2);
    y1 = y2 + 10;
  }
   

  // ======================

  var audioCtx = getAudioContext();
  var source = audioCtx.createMediaElementSource(data.control.audio);
  data.fft.setInput(source);
  source.connect(audioCtx.destination);
  // data.fft.analyser.connect(audioCtx.destination);

  if (!data.no_mic) {
    data.mic = new p5.AudioIn();
    data.mic_fft.setInput(data.mic);
    data.control.toggleMic(data.use_mic);
  } else {
    data.use_mic = false;
    data.control.showNoMicDialog();
  }

  if (!data.input_pane) {
    jQuery('#input_pane').hide();
  }
}

function draw() {
  background(255);

   top_model.getValue();
  
  if (data.show_vspec) {
    vspec.onDraw();
  }

}

function Control() {
  var self = this;
  this.mic_icon = jQuery("#mic_icon");
  this.audio_pane = jQuery("#audio_pane");
  this.audio = jQuery('#myAudio')[0];
  this.mic_on = false;

  this.pitch_name = jQuery("#pitch_name");
  this.pitch_pure = jQuery("#pitch_pure");

  jQuery('input').on('change', function(e) {
    var path = URL.createObjectURL(jQuery('#audio_file')[0].files[0]);
    self.audio.src = path;
    self.toggleAudio(true);
  });

  this.togglePlay = function() {
    data.lock_spec = !data.lock_spec;
    if (data.use_mic) {
      this.toggleMic(!data.lock_spec);
    } else {
      this.toggleAudio(!data.lock_spec);
    }
  };

  this.toggleMic = function(enable) {
    if (data.no_mic) {
      data.use_mic = false;
      return;
    }
    this.mic_on = enable;
    if (enable) {
      data.use_mic = true;
      data.mic.start();
      this.mic_icon.addClass('on');
      this.toggleAudio(false);
      data.lock_spec = false;
    } else {
      data.mic.stop();
      this.mic_icon.removeClass('on');
    }
  };
  this.toggleAudio = function(enable) {
    if (enable) {
      data.use_mic = false;
      this.audio.play();
      this.toggleMic(false);
      this.audio_pane.addClass('on');
      data.lock_spec = false;
    } else {
      this.audio.pause();
      this.audio_pane.removeClass('on');
    }
  };

  this.onClick = function() {
    if (data.show_spec && spec.mouseInRect()) {
      self.togglePlay();
    } else if (data.show_vspec && vspec.mouseInRect()) {
      self.togglePlay();
    } else if (data.show_piano && piano.mouseInRect()) {
      piano.playPitch(piano.mouseToPitch());
    }
  };

  this.onDraw = function() {
    if (data.use_mic && !this.audio.paused) {
      this.toggleAudio(true);
    }
  };

  this.drawText = function() {
    // Draw text.
    {

      var str = '';
      if (data.pitch) {
        str += data.pitch[data.pitch_name];
      }
      this.pitch_name.text(str);
      str = '';
      if (data.max_top) {
        str += data.max_top.getPureName();
      }
      this.pitch_pure.text(str);
    }
  };

  this.showNoMicDialog = function() {
    window.alert(Locale.get('no_mic'));
  }

  this.mic_icon.click(function() {
    self.toggleMic(!self.mic_on);
    if (self.mic_on) {
      data.lock_spec = false;
    }
  });

  var range_pane = jQuery('#range_pane');
  this.ranges = [];
  for (var range_name in RANGE_DATA) {
    range_pane.append('<span id="' + range_name + '">' + getRange(range_name).name + '</span>');
    this.ranges.push(jQuery('#' + range_name));
  }
  for (var i = 0; i < this.ranges.length; i++) {
    var range_span = this.ranges[i];
    range_span.click(function() {
      var name = this.id;
      data.range_name = name;
      self.updateRangePane();
    });
  }

  this.updateRangePane = function() {
    for (var i = 0; i < self.ranges.length; i++) {
      var range_span = self.ranges[i];
      if (range_span.attr('id') == data.range_name) {
        range_span.addClass('on');
      } else {
        range_span.removeClass('on');
      }
    }
  };

  this.updateRangePane();
}

function Top(index) {
  this.total_eng = -1;
  this.left_top = null;
  this.left_low_eng = 0;
  this.overtone_count = 1;
  this.valid = true;
  this.pure = 0;



  this.init(index);
}


Top.prototype.init = function(index) {
  this.right_index = index;
  this.eng = data.spectrum[index];
  this.ori_eng = data.ori_spectrum[index];
  this.left_index = this.extend(index, -1);
  this.right_index = this.extend(index, 1);
  this.index = (this.left_index + this.right_index) / 2;
  if (this.eng < data.min_eng) this.valid = false;
};

Top.prototype.extend = function(index, delta) {
  while (true) {
    index += delta;
    if (index < 0 || index > data.x_max) return index - delta;
    var eng = data.spectrum[index];
    if (eng > this.eng) {
      return index - delta;
    } else if (eng + data.top_eng_range < this.eng) {
      return index - delta;
    }
  }
};

Top.prototype.calculateParent = function(i) {
  if (i == 1) return true;
  var parent_index = this.index * i;
  if (parent_index > data.x_max) {
    return false;
  }
  var parent = data.getTop(this.left_index * i, Math.ceil(this.right_index * i));

  if (!parent || parent.eng < data.accept_eng) {
    return false;
  }
  this.parents[i] = parent;
  this.overtone_count++;
  if (i < 4 || parent == data.highest_top) {
    this.left_index = Math.max(this.left_index, (parent.left_index - 1) / i);
    this.right_index = Math.min(this.right_index, (parent.right_index + 1) / i);
  }
  this.max_oevrtone_ori_eng = Math.max(this.max_oevrtone_ori_eng, parent.ori_eng);
  this.addTotalEng(parent.eng);
  return true;
};

Top.prototype.calculate = function(highest_index) {
  if (this.total_eng > 0) return 0;
  this.addTotalEng(this.eng);
  this.self_total_eng = this.total_eng;
  this.parents = [];
  this.overtone_count = 1;
  this.max_oevrtone_ori_eng = 0;
  this.accept_eng = 0;
  this.left_index--;
  this.right_index++;
  this.calculateParent(highest_index);
  for (var i = 2; i < 50; ++i) {
    if (i != highest_index) {
      if (!this.calculateParent(i)) break;
    }
  }
  this.index = (this.left_index + this.right_index) / 2;
  return this.total_eng;
};

Top.prototype.getAvgTotalEng = function() {
  return this.total_eng / this.overtone_count;
}

Top.prototype.getPure = function() {
  return this.total_eng / this.self_total_eng;
};

Top.prototype.getPureName = function() {
  var pure = this.getPure();
  var freq = data.indexToFreq(this.index);
  if (freq > 900 && pure > -8) {
    return Locale.get('head_voice');
  }
  if (pure < 1.5) {
    return Locale.get('super_fake_voice');
  } else if (pure < 3) {
    return Locale.get('pure_fake_voice');
  } else if (pure < 4) {
    return Locale.get('fake_voice');
  } else if (pure < 5) {
    return Locale.get('half_fake_voice');
  } else if (pure < 6) {
    return Locale.get('mix_voice');
  } else if (pure < 20) {
    return Locale.get('modal_voice');
  } else {
    return Locale.get('pure_modal_voice');
  }
};

Top.prototype.addTotalEng = function(eng) {
  var sone = Math.pow(2, (eng - 40) / 10);
  this.total_eng += sone;
};

var top_model = {


  getValue: function() {
    data.tops = new Array(data.fft_size);

    if (!data.lock_spec) {
      var fft = data.use_mic ? data.mic_fft : data.fft;
      data.spectrum = fft.analyze('db');
      data.sample_rate = fft.input.context.sampleRate;
      var max_eng = 120;
      data.highest_index = 0;
      data.highest_eng = 0;
      data.ori_spectrum = new Array(data.spectrum.length);
      for (var i = 0; i < data.x_max; ++i) {
        var value = data.spectrum[i] + 140;
        data.ori_spectrum[i] = value;
        data.spectrum[i] = data.equal_loud.adjust(i, value);
        if (value > data.highest_eng) {
          data.highest_eng = value;
          data.highest_index = i;
        }
        max_eng = Math.max(max_eng, data.ori_spectrum[i]);
      }
      data.max_eng = max_eng;
      data.top_eng_range = (data.max_eng - data.min_eng) * data.top_eng_range_rate;
    }
  },

  guessPitch: function() {
    var spectrum = data.spectrum;
    data.pitch = null;
    data.max_top = null;
    // ====================== first loop
    {
      var last_up_index = 0;
      var low_eng = 1000;
      var last_top = null;

      for (var i = 0; i < spectrum.length; i++) {
        var eng = spectrum[i];

        if (i > 0) {
          if (eng > spectrum[i - 1]) {
            last_up_index = i;
          } else if (eng < spectrum[i - 1] && last_up_index > 0) {
            var top = null;
            if (!data.tops[last_up_index]) {
              top = new Top(last_up_index);
              if (!top.valid) {
                top = null;
              }
            }
            // check invalid top
            if (top) {
              while (last_top) {
                if (last_top.index * 2 <= top.index) break;

                if (top.eng - low_eng >
                  data.fake_top_rate * (last_top.eng - low_eng)) {
                  // delete last top.
                  for (var k = last_top.left_index; k <= last_top.right_index; ++k) {
                    data.tops[k] = null;
                  }
                  low_eng = Math.min(low_eng, last_top.left_low_eng);
                  last_top = last_top.left_top;
                } else if (last_top.eng - low_eng >
                  data.fake_top_rate * (top.eng - low_eng)) {
                  top = null;
                  break;
                } else {
                  break;
                }
              }
            }


            if (top) {
              top.left_low_eng = low_eng;
              top.left_top = last_top;
              for (var k = top.left_index; k <= top.right_index; ++k) {
                data.tops[k] = top;
              }
              last_up_index = 0;
              last_top = top;
              low_eng = top.eng;
            }
            if (eng < low_eng) {
              low_eng = eng;
            }

          }
        }

      }
    }

    // Guess pitch.
    var max_top = null; {
      data.highest_top = data.tops[data.highest_index];
      if (data.highest_top) {
        max_top = data.highest_top;
        var max_top_eng = data.highest_top.calculate(1);
        var highest_total_eng = data.highest_top.total_eng;
        var highest_overtone_count = data.highest_top.overtone_count;

        data.accept_eng = 0; //(data.highest_top.eng - data.min_eng) / 4 + data.min_eng;
        for (var i = 2; i < 10; ++i) {
          var top = data.getTop(data.highest_top.left_index / i, data.highest_top.right_index / i);
          if (!top) continue;
          if (top.index <= 6) break;
          var total_eng = top.calculate(i);
          if (top.overtone_count <= highest_overtone_count) continue;
          if (top.total_eng < highest_total_eng) continue;

          if (total_eng > max_top_eng) {
            max_top_eng = total_eng;
            max_top = top;
          }
        }
      }

      if (max_top) {
        var freq = max_top.index * data.sample_rate / 2 / data.fft_size;
        data.pitch = getPitch(freq);
        data.max_top = max_top;
      }
    }
  },
};



function Buffer(width, height) {
  this.scale = data.image_scale;
  this.width = width * this.scale;
  this.height = height * this.scale;
  this.buffer = createImage(this.width, this.height);
  this.buffer.loadPixels();
}

Buffer.prototype.loadPixels = function() {
  this.buffer.loadPixels();
}

Buffer.prototype.updatePixels = function() {
  this.buffer.updatePixels();
}

Buffer.prototype.fillPoint = function(x, y, color) {
  var scale = this.scale;
  x *= scale;
  y *= scale;
  for (var i = 0; i < scale; ++i) {
    for (var j = 0; j < scale; j++) {
      this.buffer.set(x + i, y + j, color);
    }
  }
}

var vspec = {
  x_min: 0,
  x_max: 100,
  y_min: 0,
  y_max: 100,
  x_scale: 1,
  y_scale: 1,
  x_current: 0,


  fillData: function() {
    if (!data.lock_spec) {
      var spectrum = data.ori_spectrum;
      for (var i = 0; i < spectrum.length; i++) {
        var ii = i * this.y_scale;
        if (ii > this.height) break;
        var eng = spectrum[i];
        
        if(eng!=0)
            //console.log("eng:",eng)
        var pitch_color = this.engToColor(eng);

        this.buffer.fillPoint(this.x_current, this.height - ii, pitch_color);
      }
      this.x_current += this.x_scale;
      if (this.x_current >= this.width) {
        this.x_current = 0;
      }
      this.buffer.updatePixels();
    }
  },

  onDraw: function() {
    this.fillData();

    stroke(data.border_color);
    fill(data.spec_background_color);
    //rect(this.x_min, this.y_min, this.width, this.height);
    var x_break = this.x_current * this.buffer.scale;
    image(this.buffer.buffer, x_break, 0, this.buffer.width - x_break, this.buffer.height,
      this.x_min, this.y_min, this.width - this.x_current, this.height);
    if (this.x_current > 0) {
      image(this.buffer.buffer, 0, 0, x_break, this.buffer.height,
        this.x_max - this.x_current, this.y_min, this.x_current, this.height);
    }

  },

  // drawPitch: function(pitch, color_name) {
    // if (!pitch) return;
    // this.drawPitchLine(pitch.value, pitch[data.pitch_name], color_name);
  // },
  // drawPitchLine: function(freq, pitch_text, color_name) {
    // var y = this.freqToY(freq);
    // if (y < this.y_min) return;

    // data.setAlphaColor(color_name);
    // line(this.x_min, y, this.x_max, y);
    // data.setColor(color_name);
    // text(pitch_text, this.x_max + 3, y + 7);
  // },
  // drawMixPitch: function(pitch, both) {
    // if (!pitch) return;
    // this.drawMixPitchLine(pitch.value, both);
  // },
  // drawMixPitchLine: function(freq, both) {
    // var y = this.freqToY(freq);
    // if (y < this.y_min) return;

    // var x = this.x_max + 5;
    // data.setAlphaColor('man_pitch_color');
    // line(this.x_min, y, this.x_max, y);
    // data.setColor('man_pitch_color');
    // this.drawTri(x, y);
    // x +=  13;
    // text(Locale.get('man'), x, y + 6);
    // if (both) {
      // y -= 1;
      // x += 24;
      // data.setAlphaColor('woman_pitch_color');
      // line(this.x_min, y, this.x_max, y);
      // data.setColor('woman_pitch_color');
      // this.drawTri(x, y);
      // x +=  13;
      // text(Locale.get('woman'), x, y + 6);
    // }
  // },
  // drawTri: function(x, y) {
    // var xx = 10;
    // var yy = 7;
    // triangle(x, y, x + xx, y + yy, x + xx, y - yy);
  // },


  init: function(x1, x2, y1, y2) {
    this.x_min = x1;
    this.x_max = x2;
    this.y_min = y1;
    this.y_max = y2;
    //this.y_scale *= data.fft_scale;
    this.width = this.x_max - this.x_min;
    this.height = this.y_max - this.y_min;
    data.x_max = this.height;
    data.y_max = this.height / this.y_scale;
    this.buffer = new Buffer(this.width, this.height);
    for (var x = 0; x < this.width; x++) {
      for (var y = 0; y < this.height; y++) {
        this.buffer.fillPoint(x, y, color('#000'));
      }
    }
    var cache = new Array(101);
    var r = red(data.fill_color);
    var g = green(data.fill_color);
    var b = blue(data.fill_color);
    for (var db = 0; db <= 100; db++) {
      var percent = db / 100;
      cache[db] = color(r * percent, g * percent, b * percent);
    }
    this.color_cache = cache;
  },

  // yToFreq: function(y) {
    // y -= this.y_min;
    // y = this.height - y;
    // return map(y / this.y_scale, 0, data.fft_size, 0, data.sample_rate / 2);
  // },
  // freqToY: function(freq) {
    // var y = map(freq, 0, data.sample_rate / 2, 0, data.fft_size);
    // return this.y_max - y * this.y_scale;
  // },
  engToColor: function(eng) {
    var percent = map(eng, data.min_eng, data.max_eng - data.eng_delta, 0, 100);
    percent = Math.min(100, Math.round(percent));
    percent = Math.max(0, percent);
    return this.color_cache[percent];
  },
  mouseInRect: function() {
    return mouseX > this.x_min && mouseX < this.x_max &&
      mouseY > this.y_min && mouseY < this.y_max
  },
}



var data = {
  mic: null,
  audio: null,
  fft: null,
  spectrum: null,
  tops: null,

  input_pane: true,
  show_spec: true,
  show_vspec: true,
  show_piano: true,
  show_pitch_on_spec: false,
  use_mic: true,
  no_mic: false,
  debug: false,

  sample_rate: 44100,
  fft_size: 1024,
  max_db: 140,

  lock_spec: false,

  width: 800,
  padding_right: 50,
  x_max: 1000,
  avg_spec: 0,
  min_eng: 60,
  max_eng: 140,
  eng_delta: 20,
  image_scale: 1,
  fft_scale: 1,
  smooth: 0.01,
  fps: 30,

  overtone_count: 5,
  overtone_accept_count: 3,
  overtone_accept_percent: 0.99,
  mix_overtone: false,
  top_eng_range_rate: 0.03,
  fake_top_rate: 4,

  pitch_name: 'inter',
  range_name: 'man_high',

  color_alpha: 220,
  spec_background_color: '#000',
  fill_color: '#4FC3F7',
  stroke_color: '#0277BD',
  spec_text_color: '#212121',

  vspec_pitch_color: '#F44336',
  vspec_color_cache: true,

  pitch_color: '#F44336',
  range_color: 'rgba(33, 150, 243, 0.2)',
  man_pitch_color: '#64B5F6',
  woman_pitch_color: '#E64A19',

  overtone_color: '#F57F17',
  mouse_pitch_color: '#F9A825',
  mouse_pitch_piano_color: '#F9A825',
  line_color: '#9E9D24',
  line_value: 4000,

  border_color: '#212121',

  piano_white_color: '#FFFFFF',
  piano_black_color: '#000000',

  getTop: function(left, right) {
    var max_eng = 0;
    var max_top = null;
    left = Math.floor(left);
    right = Math.ceil(right);
    for (var i = left; i <= right; ++i) {
      var top = this.tops[i];
      if (top && top.eng > max_eng) {
        max_eng = top.eng;
        max_top = top;
      }
    }
    return max_top;
  },
  indexToFreq: function(i) {
    return map(i, 0, data.fft_size, 0, data.sample_rate / 2);
  },
  setColor: function(color_name) {
    stroke(this[color_name]);
    fill(this[color_name]);
  },
  setAlphaColor: function(color_name) {
    stroke(this[color_name + '_alpha']);
    fill(this[color_name + '_alpha']);
  },

  init: function() {
    var screen_width = screen.width;
    if (screen_width < this.width) {
      this.width = screen_width - this.padding_right;
      vspec.x_scale = 1;
      this.show_spec = false;
    }

    if (!window.navigator.getUserMedia) {
      this.no_mic = true;
    }

    var is_safari = /Safari/.test(navigator.userAgent) && /Apple Computer/.test(navigator.vendor);

    if (!is_safari) {
      this.fft_size = 2048;
      this.fft_scale = 1;
    }

    var prmstr = window.location.search.substr(1);
    if (prmstr != null && prmstr !== "") {

      var params = {};
      var prmarr = prmstr.split("&");
      for (var i = 0; i < prmarr.length; i++) {
        var tmparr = prmarr[i].split("=");
        var name = tmparr[0];
        var value = tmparr[1];
        if (value == 'false') {
          value = false
        } else if (jQuery.isNumeric(value)) {
          value = Number(value);
        }
        if (name.localeCompare('backup_pitch') == 0) {
          value = getPitch(value);
        }
        this[name] = value;
      }
    }
    this.line_value /= this.fft_size / 1024 * this.fft_scale;

    // handle colors.
    for (var key in this) {
      if (key.endsWith('_color')) {
        var c_name = this[key];
        if (c_name.length == 6) {
          c_name = '#' + c_name;
        }
        var c = color(c_name);
        this[key] = c;
        this[key + '_alpha'] = color(red(c), green(c), blue(c), this.color_alpha);
      }
    }
  },
};