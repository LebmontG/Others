classdef Policy < handle
    properties
        fis
    end

    methods
        
        function self = Policy()
            self.fis=newfis('fuzz_tank');
            self.fis=addvar(self.fis,'input','e',[-20,20]);
            self.fis=addmf(self.fis,'input',1,'NB','gaussmf',[2,-4]);
            self.fis=addmf(self.fis,'input',1,'NS','gaussmf',[2,-2]);
            self.fis=addmf(self.fis,'input',1,'Z','gaussmf',[2,0]);
            self.fis=addmf(self.fis,'input',1,'PS','gaussmf',[2,2]);
            self.fis=addmf(self.fis,'input',1,'PB','gaussmf',[2,4]);
            self.fis=addvar(self.fis,'output','u',[0,8]);          %Parameter u
            self.fis=addmf(self.fis,'output',1,'NB','gaussmf',[2,-8]);
            self.fis=addmf(self.fis,'output',1,'NS','gaussmf',[2,-4]);
            self.fis=addmf(self.fis,'output',1,'Z','gaussmf',[2,0]);
            self.fis=addmf(self.fis,'output',1,'PS','gaussmf',[2,2]);
            self.fis=addmf(self.fis,'output',1,'PB','gaussmf',[2,8]);
            rulelist=[1 1 1 1;2 2 1 1;3 3 1 1;4 4 1 1;5 5 1 1];
            self.fis=addrule(self.fis,rulelist);
            self.fis=setfis(self.fis,'DefuzzMethod','bisector');
        end
        
        function action=action(self,observation)
            state=[observation.targetHeight-observation.agent.H];
            if state>20
                state=20
            end
            if state<-20
                state=-20
            end
            action=evalfis(state,self.fis);
        end
        
        
    end
end