% image1 = imread('room1.jpg');
% image2 = imread('room2.jpg');
% image1 = imread('yard1.jpg');
% image2 = imread('yard2.jpg');
image1 = imread('apartment1.jpg');
image2 = imread('apartment2.jpg');

image1 = rgb2gray(image1);
image2 = rgb2gray(image2);

points1 = detectSURFFeatures(image1);
points2 = detectSURFFeatures(image2);

[features1, valid_points1] = extractFeatures(image1, points1);
[features2, valid_points2] = extractFeatures(image2, points2);

indexPairs = matchFeatures(features1, features2);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

% figure; showMatchedFeatures(image1,image2,matchedPoints1,matchedPoints2);

x1 = zeros(4,1);
y1 = zeros(4,1);
x2 = zeros(4,1);
y2 = zeros(4,1);
bestX1 = zeros(4,1);
bestY1 = zeros(4,1);
bestX2 = zeros(4,1);
bestY2 = zeros(4,1);

n = 0;
thresh = 5; % 'good' point threshold
grade = 0;
bestGrade = 0;
bestH = zeros(3,3);
numIterations = 5000; % nunmber of RANSAC iterations

while n < numIterations
    % Getting 4 random points and their (x,y) values
    pointIndices = zeros(4,1);
    
    for i = 1 : size(pointIndices,1)
        pointIndices(i) = randi([1 size(matchedPoints1,1)]);
    end
    
    for i = 1 : 4
        x1(i) = matchedPoints1.Location(pointIndices(i),1);
        y1(i) = matchedPoints1.Location(pointIndices(i),2);
        x2(i) = matchedPoints2.Location(pointIndices(i),1);
        y2(i) = matchedPoints2.Location(pointIndices(i),2);
    end
    
    % Normalize points
    x1 = (x1 - mean(x1) / std(x1));
    y1 = (y1 - mean(y1) / std(y1));
    x2 = (x2 - mean(x2) / std(x2));
    y2 = (y2 - mean(y2) / std(y2));
    
    % (x1, y1) = (x, y)
    % (x2, y2) = (x', y')
    % Building the point matrix
    A = zeros(8,9, 'double');
    for i = 1 : 4
        index = i * 2 - 1;
        A(index,:) = [x1(i) y1(i) 1 0 0 0 -x2(i)*x1(i) -x2(i)*y1(i) -x2(i)];
    end
    for i = 1 : 4
        index = i * 2;
        A(index,:) = [0 0 0 x1(i) y1(i) 1 -y2(i)*x1(i) -y2(i)*y1(i) -y2(i)];
    end
    
    % Get smallest eigenvector to use as H
    [H, ~] = eigs(A.'*A,1,'sm');
    p1 = matchedPoints1.Location;
    p2 = matchedPoints2.Location;
    
    % Reshape H to a 3x3 matrix
    HP = reshape(H,3,3)';
    
    % Use H to calculate all remaining matching points
    p3 = zeros(size(p1,1),2);
    for i = 1 : size(p2,1)
        pt = p1(i,:)';
        pt(3) = 1;
        npt = HP * pt;
        npt = npt ./ npt(3,1);
        p3(i,:) = [npt(1,1), npt(2,1)];
    end
    
    % Calculate the distance between the actual and calculated key points
    xDiff = abs(p3(:,1) - p2(:,1));
    yDiff = abs(p3(:,2) - p2(:,2));
    distance = sqrt(xDiff.^2 + yDiff.^2);
    
    % Check if points are within threshold value
    match = distance < thresh;
    
    % 'Grade' the homography by calculating what percent are 'good' matches
    grade = mean(match);
    
    % Update best 'grade' and best H if grade greater than current best
    if (grade > bestGrade)
        bestGrade = grade;
        bestH = H;
    end
    n = n + 1;
end
% disp(bestGrade);

% Get the best H as a 3x3
M = reshape(bestH,3,3)';
sizeX = size(image2,1);
sizeY = size(image2,2);

final = zeros(sizeX*2,sizeY*2,3,'uint8');

% Place image 1 in the certer of an image 4 times larger, for stitching
offsetX = floor(sizeX/2);
offsetY = floor(sizeY/2);

% Add red channel of grayscale image 1 to final image
final(offsetX:3*offsetX-1,offsetY:3*offsetY-1,1) = image1(:,:);

for i = 1 - offsetY : sizeY + offsetY
    for j = 1 - offsetX : sizeX + offsetY
        % Calculate point after transformation in image 2
        new = M * [i; j; 1];
        newX = round(new(2)./new(3));
        newY = round(new(1)./new(3));
        
        % If point in in bounds, retreive and set the G and B channels
        if (newX > 0 && newY > 0 && newX <= sizeX && newY <= sizeY)
            final(j + offsetX,i + offsetY, 3) = image2(newX, newY);
            final(j + offsetX,i + offsetY, 2) = image2(newX, newY);
        end
    end
end

figure;
imshow(final);



