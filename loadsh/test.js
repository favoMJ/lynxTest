import _ from 'lodash';
function square(n) {
    return n * n;
}

var wrapped = _([1, 2, 3]);

// 返回未包装的值
wrapped.reduce(_.add);
// => 6

// 返回链式包装的值
var squares = wrapped.map(square);

_.isArray(squares);
// => false

_.isArray(squares.value());
// => true
